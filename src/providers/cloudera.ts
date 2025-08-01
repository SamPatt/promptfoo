import { getEnvString } from '../envars';
import { OpenAiChatCompletionProvider } from './openai/chat';

import type { ProviderOptions } from '../types/providers';
import type { OpenAiCompletionOptions } from './openai/types';

type ClouderaAiCompletionOptions = OpenAiCompletionOptions & {
  domain?: string;
  namespace?: string;
  endpoint?: string;
};

type ClouderaAiProviderOptions = ProviderOptions & {
  config: ClouderaAiCompletionOptions;
};

export class ClouderaAiChatCompletionProvider extends OpenAiChatCompletionProvider {
  constructor(modelName: string, providerOptions: ClouderaAiProviderOptions) {
    // https://docs.cloudera.com/machine-learning/cloud/ai-inference/topics/ml-caii-openai-inference-protocol-using-curl.html
    const domain = providerOptions.config?.domain || getEnvString('CDP_DOMAIN');
    const namespace = providerOptions.config?.namespace || 'serving-default';
    const endpoint = providerOptions.config?.endpoint || modelName;

    super(modelName, {
      ...providerOptions,
      config: {
        ...providerOptions.config,
        apiKeyEnvar: 'CDP_TOKEN',
        apiBaseUrl: `https://${domain}/namespaces/${namespace}/endpoints/${endpoint}/v1`,
      },
    });
  }
}
