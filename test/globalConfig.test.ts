import * as fs from 'fs';

import yaml from 'js-yaml';

import type {
  readGlobalConfig,
  writeGlobalConfig,
  writeGlobalConfigPartial,
} from '../src/globalConfig/globalConfig';

// Helper function to validate UUID format
function isValidUUID(uuid: string): boolean {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidRegex.test(uuid);
}

// Clear any existing mocks
jest.unmock('../src/globalConfig/globalConfig');

jest.mock('fs', () => ({
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
  statSync: jest.fn(),
  readdirSync: jest.fn(),
  existsSync: jest.fn(),
  mkdirSync: jest.fn(),
}));

jest.mock('proxy-agent', () => ({
  ProxyAgent: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('glob', () => ({
  globSync: jest.fn(),
}));

jest.mock('../src/database');

describe('Global Config', () => {
  let globalConfig: {
    readGlobalConfig: typeof readGlobalConfig;
    writeGlobalConfig: typeof writeGlobalConfig;
    writeGlobalConfigPartial: typeof writeGlobalConfigPartial;
  };

  beforeEach(async () => {
    jest.clearAllMocks();
    await jest.isolateModules(async () => {
      globalConfig = await import('../src/globalConfig/globalConfig');
    });
  });

  const mockConfig = { id: 'test-id', account: { email: 'test@example.com' } };

  describe('readGlobalConfig', () => {
    describe('when config file exists', () => {
      beforeEach(() => {
        jest
          .mocked(fs.existsSync)
          .mockImplementation(
            (path) =>
              path.toString().includes('promptfoo.yaml') || path.toString().includes('.promptfoo'),
          );
        jest.mocked(fs.readFileSync).mockReturnValue(yaml.dump(mockConfig));
      });

      it('should read and parse the existing config file', () => {
        const result = globalConfig.readGlobalConfig();
        expect(fs.readFileSync).toHaveBeenCalledWith(
          expect.stringContaining('promptfoo.yaml'),
          'utf-8',
        );
        expect(result).toEqual(mockConfig);
      });

      it('should handle empty config file by returning config with generated ID', () => {
        jest.mocked(fs.readFileSync).mockReturnValue('');

        const result = globalConfig.readGlobalConfig();

        expect(result).toEqual({ id: expect.any(String) });
        expect(result.id).toBeDefined();
        expect(typeof result.id).toBe('string');
        expect(isValidUUID(result.id!)).toBe(true);
      });

      it('should generate and save ID when existing config lacks an ID', () => {
        const configWithoutId = { account: { email: 'test@example.com' } };
        jest.mocked(fs.readFileSync).mockReturnValue(yaml.dump(configWithoutId));
        jest.mocked(fs.writeFileSync).mockImplementation();

        const result = globalConfig.readGlobalConfig();

        expect(result.id).toBeDefined();
        expect(typeof result.id).toBe('string');
        expect(isValidUUID(result.id!)).toBe(true);
        expect(result.account).toEqual(configWithoutId.account);
        // Should have written the config with the new ID
        expect(fs.writeFileSync).toHaveBeenCalledWith(
          expect.stringContaining('promptfoo.yaml'),
          expect.stringContaining(`id: ${result.id}`),
        );
      });
    });

    describe('when config file does not exist', () => {
      beforeEach(() => {
        jest.mocked(fs.existsSync).mockReturnValue(false);
        jest.mocked(fs.writeFileSync).mockImplementation();
        jest.mocked(fs.mkdirSync).mockImplementation();
      });

      it('should create new config directory and file with generated UUID', () => {
        const result = globalConfig.readGlobalConfig();

        expect(fs.existsSync).toHaveBeenCalledTimes(2);
        expect(fs.mkdirSync).toHaveBeenCalledWith(expect.any(String), { recursive: true });
        expect(fs.writeFileSync).toHaveBeenCalledTimes(1);
        expect(fs.writeFileSync).toHaveBeenCalledWith(
          expect.stringContaining('promptfoo.yaml'),
          expect.any(String),
        );
        expect(result).toEqual({ id: expect.any(String) });
        expect(result.id).toBeDefined();
        expect(typeof result.id).toBe('string');
        expect(isValidUUID(result.id!)).toBe(true);
      });
    });

    describe('error handling', () => {
      it('should throw error if config file is invalid YAML', () => {
        jest.mocked(fs.existsSync).mockReturnValue(true);
        jest.mocked(fs.readFileSync).mockReturnValue('invalid: yaml: content:');

        expect(() => globalConfig.readGlobalConfig()).toThrow(/bad indentation of a mapping entry/);
      });
    });
  });

  describe('writeGlobalConfig', () => {
    it('should write config to file in YAML format', () => {
      globalConfig.writeGlobalConfig({
        ...mockConfig,
      });

      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('promptfoo.yaml'),
        expect.stringContaining('account:'),
      );
    });
  });

  describe('writeGlobalConfigPartial', () => {
    beforeEach(() => {
      // Setup initial config
      jest.mocked(fs.existsSync).mockReturnValue(true);
      jest.mocked(fs.readFileSync).mockReturnValue(
        yaml.dump({
          account: { email: 'old@example.com' },
          cloud: { apiKey: 'old-key', apiHost: 'old-host' },
        }),
      );
    });

    it('should merge new config with existing config', () => {
      const partialConfig = {
        account: { email: 'new@example.com' },
      };

      globalConfig.writeGlobalConfigPartial(partialConfig);

      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('promptfoo.yaml'),
        expect.stringMatching(/email: new@example\.com.*apiKey: old-key/s),
      );
    });

    it('should remove keys when value is falsy', () => {
      const partialConfig = {
        cloud: undefined,
      };

      globalConfig.writeGlobalConfigPartial(partialConfig);

      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('promptfoo.yaml'),
        expect.not.stringContaining('apiKey: old-key'),
      );
    });

    it('should update specific keys while preserving others', () => {
      const partialConfig = {
        cloud: { apiKey: 'new-key' },
      };

      globalConfig.writeGlobalConfigPartial(partialConfig);

      const writeCall = jest.mocked(fs.writeFileSync).mock.calls[1][1] as string;
      const writtenConfig = yaml.load(writeCall) as any;

      expect(writtenConfig.cloud.apiKey).toBe('new-key');
      expect(writtenConfig.account.email).toBe('old@example.com');
    });
  });
});
