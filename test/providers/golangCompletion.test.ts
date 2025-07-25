import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';

import { getCache, isCacheEnabled } from '../../src/cache';
import { GolangProvider } from '../../src/providers/golangCompletion';

jest.mock('child_process');
jest.mock('../../src/cache');
jest.mock('fs');
jest.mock('path');

jest.mock('../../src/util', () => {
  const actual = jest.requireActual('../../src/util');
  return {
    ...actual,
    parsePathOrGlob: jest.fn(() => ({
      extension: 'go',
      functionName: undefined,
      isPathPattern: false,
      filePath: '/absolute/path/to/script.go',
    })),
  };
});

describe('GolangProvider', () => {
  const mockExec = jest.mocked(exec);
  const mockGetCache = jest.mocked(getCache);
  const mockIsCacheEnabled = jest.mocked(isCacheEnabled);
  const mockReadFileSync = jest.mocked(fs.readFileSync);
  const mockResolve = jest.mocked(path.resolve);
  const mockMkdtempSync = jest.mocked(fs.mkdtempSync);
  const mockExistsSync = jest.mocked(fs.existsSync);
  const mockRmSync = jest.mocked(fs.rmSync);
  const mockReaddirSync = jest.mocked(fs.readdirSync);
  const mockCopyFileSync = jest.mocked(fs.copyFileSync);
  const mockMkdirSync = jest.mocked(fs.mkdirSync);
  const mockDirname = jest.mocked(path.dirname);
  const mockJoin = jest.mocked(path.join);
  const mockRelative = jest.mocked(path.relative);

  beforeEach(() => {
    jest.clearAllMocks();
    jest.resetAllMocks();

    // Reset all mocks to default behavior
    mockGetCache.mockResolvedValue({
      get: jest.fn(),
      set: jest.fn(),
    } as never);
    mockIsCacheEnabled.mockReturnValue(false);

    // Mock readFileSync to return content for all files
    mockReadFileSync.mockImplementation((filePath: fs.PathOrFileDescriptor) => {
      const pathStr = filePath.toString();
      if (pathStr.includes('wrapper.go')) {
        return 'package main\n// Mock wrapper.go content';
      }
      if (pathStr.includes('script.go')) {
        return 'package main\n// Mock script.go content';
      }
      return 'mock file content';
    });

    const mockParsePathOrGlob = jest.requireMock('../../src/util').parsePathOrGlob;
    mockParsePathOrGlob.mockImplementation((basePath: string, runPath: string) => {
      if (!basePath && runPath === 'script.go') {
        return {
          filePath: 'script.go',
          functionName: undefined,
          isPathPattern: false,
          extension: 'go',
        };
      }
      return {
        filePath: runPath.replace(/^(file:\/\/|golang:)/, '').split(':')[0],
        functionName: runPath.includes(':') ? runPath.split(':')[1] : undefined,
        isPathPattern: false,
        extension: 'go',
      };
    });

    // Mock path operations
    mockResolve.mockImplementation((p: string) => {
      if (!p || typeof p !== 'string') {
        return '/absolute/path/undefined';
      }
      if (p === 'script.go') {
        return 'script.go';
      }
      if (p.includes('script.go')) {
        return '/absolute/path/to/script.go';
      }
      if (p.includes('wrapper.go')) {
        return '/absolute/path/to/wrapper.go';
      }
      return p.startsWith('/') ? p : `/absolute/path/to/${p}`;
    });

    mockRelative.mockImplementation((from: string, to: string) => {
      if (!from && to === 'script.go') {
        return 'script.go';
      }
      if (to.includes('script.go')) {
        return 'script.go';
      }
      return to;
    });

    mockDirname.mockImplementation((p: string) => {
      const paths: Record<string, string> = {
        '/absolute/path/to/script.go': '/absolute/path/to',
        '/absolute/path/to': '/absolute/path',
        '/absolute/path': '/absolute',
        '/absolute': '/',
      };
      return paths[p] || p.split('/').slice(0, -1).join('/') || '/';
    });

    mockJoin.mockImplementation((...paths: string[]) => {
      const validPaths = paths.filter((p) => p !== undefined && p !== null);
      if (validPaths.length === 0) {
        return '';
      }

      // Handle specific test cases
      if (validPaths.includes('go.mod')) {
        const basePath = validPaths.find((p) => p !== 'go.mod') || '/absolute/path/to';
        return `${basePath}/go.mod`;
      }
      if (validPaths.includes('wrapper.go')) {
        return '/tmp/golang-provider-xyz/wrapper.go';
      }

      return validPaths.join('/').replace(/\/+/g, '/');
    });

    // Mock file system operations
    mockMkdtempSync.mockReturnValue('/tmp/golang-provider-xyz');

    mockExistsSync.mockImplementation((p: fs.PathLike) => {
      const pathStr = p.toString();
      // Always return true for go.mod files and wrapper.go
      if (pathStr.includes('go.mod') || pathStr.includes('wrapper.go')) {
        return true;
      }
      return true;
    });

    // Mock directory reading to return predictable results
    mockReaddirSync.mockImplementation((p: fs.PathOrFileDescriptor) => {
      const pathStr = p.toString();
      if (pathStr.includes('nested')) {
        return [{ name: 'nested-file.go', isDirectory: () => false }] as any;
      }
      return [{ name: 'test.go', isDirectory: () => false }] as any;
    });

    // Mock directory and file operations
    mockMkdirSync.mockImplementation(() => undefined);
    mockCopyFileSync.mockImplementation(() => undefined);
    mockRmSync.mockImplementation(() => undefined);

    // Mock exec with proper async behavior
    mockExec.mockImplementation(((cmd: string, callback: any) => {
      if (!callback) {
        return {} as any;
      }

      // Use setImmediate to ensure proper async behavior
      setImmediate(() => {
        try {
          if (cmd.includes('cd') && cmd.includes('go build')) {
            callback(null, { stdout: '', stderr: '' }, '');
          } else if (cmd.includes('golang_wrapper')) {
            callback(null, { stdout: '{"output":"test output"}', stderr: '' }, '');
          } else {
            callback(new Error('test error'), { stdout: '', stderr: '' }, '');
          }
        } catch (error) {
          callback(error, { stdout: '', stderr: '' }, '');
        }
      });
      return {} as any;
    }) as any);
  });

  describe('constructor', () => {
    it('should initialize with correct properties', () => {
      const provider = new GolangProvider('script.go', {
        id: 'testId',
        config: { basePath: '/absolute/path/to' },
      });
      expect(provider.id()).toBe('testId');
    });

    it('should initialize with golang: syntax', () => {
      const provider = new GolangProvider('golang:script.go', {
        id: 'testId',
        config: { basePath: '/base' },
      });
      expect(provider.id()).toBe('testId');
    });

    it('should initialize with file:// prefix', () => {
      const provider = new GolangProvider('file://script.go', {
        id: 'testId',
        config: { basePath: '/base' },
      });
      expect(provider.id()).toBe('testId');
    });

    it('should initialize with file:// prefix and function name', () => {
      const provider = new GolangProvider('file://script.go:function_name', {
        id: 'testId',
        config: { basePath: '/base' },
      });
      expect(provider.id()).toBe('testId');
    });

    it('should handle undefined basePath and use default id', () => {
      const provider = new GolangProvider('script.go');
      expect(provider.id()).toBe('golang:script.go:default');
      expect(provider.config).toEqual({});
    });

    it('should use class id() method when no id is provided in options', () => {
      const provider = new GolangProvider('script.go:custom_function');
      expect(provider.id()).toBe('golang:script.go:custom_function');
    });

    it('should allow id() override and later retrieval', () => {
      const provider = new GolangProvider('script.go');
      expect(provider.id()).toBe('golang:script.go:default');

      const originalId = provider.id;
      provider.id = () => 'overridden-id';

      expect(provider.id()).toBe('overridden-id');

      provider.id = originalId;
      expect(provider.id()).toBe('golang:script.go:default');
    });
  });

  describe('caching', () => {
    it('should use cached result when available', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockIsCacheEnabled.mockReturnValue(true);
      const mockCache = {
        get: jest.fn().mockResolvedValue(JSON.stringify({ output: 'cached result' })),
        set: jest.fn(),
      };
      mockGetCache.mockResolvedValue(mockCache as never);

      const result = await provider.callApi('test prompt');

      expect(mockCache.get).toHaveBeenCalledWith(expect.stringContaining('golang:'));
      expect(mockExec).not.toHaveBeenCalled();
      expect(result).toEqual({ output: 'cached result' });
    });

    it('should handle cache errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockIsCacheEnabled.mockReturnValue(true);
      const mockCache = {
        get: jest.fn().mockRejectedValue(new Error('Cache error')),
        set: jest.fn(),
      };
      mockGetCache.mockResolvedValue(mockCache as never);

      await expect(provider.callApi('test prompt')).rejects.toThrow('Cache error');
    });

    it('should handle cache set errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockIsCacheEnabled.mockReturnValue(true);
      const mockCache = {
        get: jest.fn().mockResolvedValue(null),
        set: jest.fn().mockRejectedValue(new Error('Cache set error')),
      };
      mockGetCache.mockResolvedValue(mockCache as never);

      await expect(provider.callApi('test prompt')).rejects.toThrow('Cache set error');
    });

    it('should not cache results that contain errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockIsCacheEnabled.mockReturnValue(true);
      const mockCache = {
        get: jest.fn().mockResolvedValue(null),
        set: jest.fn(),
      };
      mockGetCache.mockResolvedValue(mockCache as never);

      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          if (cmd.includes('golang_wrapper')) {
            callback(null, { stdout: '{"error":"test error in result"}', stderr: '' }, '');
          } else {
            callback(null, { stdout: '', stderr: '' }, '');
          }
        });
        return {} as any;
      }) as any);

      const result = await provider.callApi('test prompt');
      expect(result).toEqual({ error: 'test error in result' });
      expect(mockCache.set).not.toHaveBeenCalled();
    });

    it('should handle circular references in vars when cache is disabled', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockIsCacheEnabled.mockReturnValue(false);

      const circularObj: any = { name: 'circular' };
      circularObj.self = circularObj;

      const spy = jest
        .spyOn(provider as any, 'executeGolangScript')
        .mockImplementation(() => Promise.resolve({ output: 'mocked result' }));

      const result = await provider.callApi('test prompt', {
        prompt: {
          raw: 'test prompt',
          label: 'test',
        },
        vars: { circular: circularObj },
      });

      expect(result).toEqual({ output: 'mocked result' });
      expect(spy).toHaveBeenCalledWith(
        'test prompt',
        expect.objectContaining({
          vars: expect.objectContaining({ circular: expect.anything() }),
        }),
        'call_api',
      );

      spy.mockRestore();
    });

    it('should execute script directly when cache is not enabled', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockIsCacheEnabled.mockReturnValue(false);
      mockGetCache.mockResolvedValue({
        get: jest.fn().mockResolvedValue(null),
        set: jest.fn(),
      } as never);

      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() =>
          callback(null, { stdout: '{"output":"direct execution result"}', stderr: '' }, ''),
        );
        return {} as any;
      }) as any);

      const result = await provider.callApi('test prompt');

      expect(result).toEqual({ output: 'direct execution result' });
      const mockCache = await mockGetCache.mock.results[0].value;
      expect(mockCache.set).not.toHaveBeenCalled();
    });
  });

  describe('cleanup', () => {
    it('should clean up on error', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          callback(new Error('test error'), { stdout: '', stderr: '' }, '');
        });
        return {} as any;
      }) as any);

      await expect(provider.callApi('test prompt')).rejects.toThrow(
        'Error running Golang script: test error',
      );

      expect(mockRmSync).toHaveBeenCalledWith(
        expect.stringContaining('golang-provider'),
        expect.objectContaining({ recursive: true, force: true }),
      );
    });
  });

  describe('API methods', () => {
    it('should call callEmbeddingApi successfully', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          callback(
            null,
            {
              stdout: cmd.includes('golang_wrapper') ? '{"embedding":[0.1,0.2,0.3]}' : '',
              stderr: '',
            },
            '',
          );
        });
        return {} as any;
      }) as any);

      const result = await provider.callEmbeddingApi('test prompt');
      expect(result).toEqual({ embedding: [0.1, 0.2, 0.3] });
    });

    it('should call callClassificationApi successfully', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          callback(
            null,
            {
              stdout: cmd.includes('golang_wrapper') ? '{"classification":"test_class"}' : '',
              stderr: '',
            },
            '',
          );
        });
        return {} as any;
      }) as any);

      const result = await provider.callClassificationApi('test prompt');
      expect(result).toEqual({ classification: 'test_class' });
    });

    it('should handle stderr output without failing', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          callback(
            null,
            {
              stdout: cmd.includes('golang_wrapper') ? '{"output":"test"}' : '',
              stderr: 'warning: some go warning',
            },
            '',
          );
        });
        return {} as any;
      }) as any);

      const result = await provider.callApi('test prompt');
      expect(result).toEqual({ output: 'test' });
    });
  });

  describe('findModuleRoot', () => {
    it('should throw error when go.mod is not found', async () => {
      mockExistsSync.mockImplementation(() => false);
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      await expect(provider.callApi('test prompt')).rejects.toThrow(
        'Could not find go.mod file in any parent directory',
      );
    });

    it('should find go.mod in parent directory', async () => {
      const checkedPaths: string[] = [];

      mockExistsSync.mockImplementation((p: fs.PathLike) => {
        const pathStr = p.toString();
        checkedPaths.push(pathStr);
        return pathStr.endsWith('/absolute/path/to/go.mod');
      });

      mockDirname.mockImplementation((p: string) => p.split('/').slice(0, -1).join('/'));

      mockResolve.mockImplementation((p: string) =>
        p.startsWith('/') ? p : `/absolute/path/to/${p}`,
      );

      mockJoin.mockImplementation((...paths: string[]) => paths.join('/').replace(/\/+/g, '/'));

      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to/subdir' },
      });

      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => callback(null, { stdout: '{"output":"test"}', stderr: '' }, ''));
        return {} as any;
      }) as any);

      const result = await provider.callApi('test prompt');
      expect(result).toEqual({ output: 'test' });

      expect(checkedPaths).toContain('/absolute/path/to/go.mod');
      expect(mockExistsSync).toHaveBeenCalledWith(
        expect.stringContaining('/absolute/path/to/go.mod'),
      );
    });
  });

  describe('script execution', () => {
    it('should handle JSON parsing errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => {
          callback(
            null,
            {
              stdout: cmd.includes('golang_wrapper') ? 'invalid json' : '',
              stderr: '',
            },
            '',
          );
        });
        return {} as any;
      }) as any);

      await expect(provider.callApi('test prompt')).rejects.toThrow('Error running Golang script');
    });

    it('should use custom function name when specified', async () => {
      const mockParsePathOrGlob = jest.requireMock('../../src/util').parsePathOrGlob;
      mockParsePathOrGlob.mockReturnValueOnce({
        filePath: '/absolute/path/to/script.go',
        functionName: 'custom_function',
        isPathPattern: false,
        extension: 'go',
      });

      const provider = new GolangProvider('script.go:custom_function', {
        config: { basePath: '/absolute/path/to' },
      });

      let executedCommand = '';
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        if (cmd.includes('golang_wrapper')) {
          executedCommand = cmd;
        }
        setImmediate(() => callback(null, { stdout: '{"output":"test"}', stderr: '' }, ''));
        return {} as any;
      }) as any);

      await provider.callApi('test prompt');
      expect(executedCommand).toContain('custom_function');
      expect(executedCommand.split(' ')[2]).toBe('custom_function');
    });

    it('should use custom go executable when specified', async () => {
      const provider = new GolangProvider('script.go', {
        config: {
          basePath: '/absolute/path/to',
          goExecutable: '/custom/go',
        },
      });

      let buildCommand = '';
      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        if (cmd.includes('go build')) {
          buildCommand = cmd;
        }
        setImmediate(() => callback(null, { stdout: '{"output":"test"}', stderr: '' }, ''));
        return {} as any;
      }) as any);

      await provider.callApi('test prompt');
      expect(buildCommand).toContain('/custom/go build');
    });

    it('should handle circular references in args', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      // Create circular reference
      const circularObj: any = { name: 'circular' };
      circularObj.self = circularObj;

      // We need to access a private method, so we'll create a spy using any type assertions
      const spy = jest
        .spyOn(provider as any, 'executeGolangScript')
        .mockImplementation(() => Promise.resolve({ output: 'mocked result' }));

      // This should not throw even with circular references
      const result = await provider.callApi('test prompt', {
        prompt: {
          raw: 'test prompt',
          label: 'test',
        },
        vars: { circular: circularObj },
      });

      // Verify the result and that executeGolangScript was called
      expect(result).toEqual({ output: 'mocked result' });
      expect(spy).toHaveBeenCalledWith(
        'test prompt',
        expect.objectContaining({
          vars: expect.objectContaining({ circular: expect.anything() }),
        }),
        'call_api',
      );

      // Restore the original implementation
      spy.mockRestore();
    });
  });

  describe('file operations', () => {
    it('should handle directory copy errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });
      mockReaddirSync.mockImplementation(() => {
        throw new Error('Directory read error');
      });

      await expect(provider.callApi('test prompt')).rejects.toThrow('Directory read error');
    });

    it('should handle file copy errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockCopyFileSync.mockImplementation(() => {
        throw new Error('File copy error');
      });

      await expect(provider.callApi('test prompt')).rejects.toThrow('File copy error');
    });

    it('should copy main.go files without transformation', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      const mainGoContent = `
        package main

        // CallApi declaration
        var CallApi = func(prompt string) string {
          return "test"
        }

        func main() {
          // Some code
        }`;

      mockReaddirSync.mockReturnValue([{ name: 'main.go', isDirectory: () => false }] as any);

      mockReadFileSync.mockImplementation((p: fs.PathOrFileDescriptor) => {
        if (p.toString().endsWith('main.go')) {
          return mainGoContent;
        }
        return 'other content';
      });

      const copiedFiles: { src: string; dest: string }[] = [];
      mockCopyFileSync.mockImplementation((src: fs.PathLike, dest: fs.PathLike) => {
        copiedFiles.push({ src: src.toString(), dest: dest.toString() });
        return undefined;
      });

      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => callback(null, { stdout: '{"output":"test"}', stderr: '' }, ''));
        return {} as any;
      }) as any);

      await provider.callApi('test prompt');

      // Verify that main.go was copied, not transformed
      expect(copiedFiles).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            src: expect.stringContaining('main.go'),
            dest: expect.stringContaining('main.go'),
          }),
        ]),
      );
    });

    it('should correctly handle nested directories in copyDir', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      // Mock a nested directory structure
      mockReaddirSync.mockImplementation((p: fs.PathOrFileDescriptor) => {
        if (p.toString().includes('nested')) {
          return [{ name: 'nested-file.go', isDirectory: () => false }] as any;
        }
        return [
          { name: 'test.go', isDirectory: () => false },
          { name: 'nested', isDirectory: () => true },
        ] as any;
      });

      // Track created directories and copied files
      const createdDirs: string[] = [];
      const copiedFiles: { src: string; dest: string }[] = [];

      mockMkdirSync.mockImplementation((p: fs.PathLike) => {
        createdDirs.push(p.toString());
        return undefined;
      });

      mockCopyFileSync.mockImplementation((src: fs.PathLike, dest: fs.PathLike) => {
        copiedFiles.push({ src: src.toString(), dest: dest.toString() });
        return undefined;
      });

      mockExec.mockImplementation(((cmd: string, callback: any) => {
        if (!callback) {
          return {} as any;
        }
        setImmediate(() => callback(null, { stdout: '{"output":"test"}', stderr: '' }, ''));
        return {} as any;
      }) as any);

      await provider.callApi('test prompt');

      // Check that nested directories were created
      expect(createdDirs).toEqual(expect.arrayContaining(['/tmp/golang-provider-xyz']));
      // Check that nested files were copied
      expect(copiedFiles).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            dest: expect.stringContaining('nested-file.go'),
          }),
        ]),
      );
    });

    it('should handle copyFileSync errors', async () => {
      const provider = new GolangProvider('script.go', {
        config: { basePath: '/absolute/path/to' },
      });

      mockReaddirSync.mockReturnValue([{ name: 'main.go', isDirectory: () => false }] as any);

      mockCopyFileSync.mockImplementation(() => {
        throw new Error('Failed to copy file');
      });

      await expect(provider.callApi('test prompt')).rejects.toThrow('Failed to copy file');
    });
  });
});
