import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import yaml from 'js-yaml';
import { getEnvString } from '../../envars';
import logger from '../../logger';
import { orderKeys } from '../json';

import type { UnifiedConfig } from '../../types';

let configDirectoryPath: string | undefined = getEnvString('PROMPTFOO_CONFIG_DIR');

export function getConfigDirectoryPath(createIfNotExists: boolean = false): string {
  const p = configDirectoryPath || path.join(os.homedir(), '.promptfoo');
  if (createIfNotExists && !fs.existsSync(p)) {
    fs.mkdirSync(p, { recursive: true });
  }
  return p;
}

export function setConfigDirectoryPath(newPath: string | undefined): void {
  configDirectoryPath = newPath;
}

export function writePromptfooConfig(
  config: Partial<UnifiedConfig>,
  outputPath: string,
  headerComments?: string[],
): Partial<UnifiedConfig> {
  const orderedConfig = orderKeys(config, [
    'description',
    'targets',
    'prompts',
    'providers',
    'redteam',
    'defaultTest',
    'tests',
    'scenarios',
  ]);
  const yamlContent = yaml.dump(orderedConfig, { skipInvalid: true });
  if (!yamlContent) {
    logger.warn('Warning: config is empty, skipping write');
    return orderedConfig;
  }

  const schemaComment = `# yaml-language-server: $schema=https://promptfoo.dev/config-schema.json`;
  const headerCommentLines = headerComments
    ? headerComments.map((comment) => `# ${comment}`).join('\n') + '\n'
    : '';

  fs.writeFileSync(outputPath, `${schemaComment}\n${headerCommentLines}${yamlContent}`);
  return orderedConfig;
}
