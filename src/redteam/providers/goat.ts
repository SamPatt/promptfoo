import chalk from 'chalk';
import dedent from 'dedent';
import { VERSION } from '../../constants';
import { renderPrompt } from '../../evaluatorHelpers';
import { getUserEmail } from '../../globalConfig/accounts';
import logger from '../../logger';
import invariant from '../../util/invariant';
import { safeJsonStringify } from '../../util/json';
import { sleep } from '../../util/time';
import { getRemoteGenerationUrl, neverGenerateRemote } from '../remoteGeneration';
import { getLastMessageContent, messagesToRedteamHistory, tryUnblocking } from './shared';

import type { Assertion, AssertionSet, AtomicTestCase } from '../../types';
import type {
  ApiProvider,
  CallApiContextParams,
  CallApiOptionsParams,
  ProviderOptions,
  ProviderResponse,
} from '../../types/providers';
import type { BaseRedteamMetadata } from '../types';
import type { Message } from './shared';

/**
 * Represents metadata for the GOAT conversation process.
 */
interface GoatMetadata extends BaseRedteamMetadata {
  redteamFinalPrompt?: string;
  stopReason: 'Grader failed' | 'Max turns reached';
  successfulAttacks?: Array<{
    turn: number;
    prompt: string;
    response: string;
  }>;
  totalSuccessfulAttacks?: number;
}

/**
 * Represents the complete response from a GOAT conversation.
 */
interface GoatResponse extends ProviderResponse {
  metadata: GoatMetadata;
}

export interface ExtractAttackFailureResponse {
  message: string;
  task: string;
}

interface GoatConfig {
  injectVar: string;
  maxTurns: number;
  excludeTargetOutputFromAgenticAttackGeneration: boolean;
  stateful: boolean;
  continueAfterSuccess: boolean;
}

export default class GoatProvider implements ApiProvider {
  readonly config: GoatConfig;
  private successfulAttacks: Array<{
    turn: number;
    prompt: string;
    response: string;
  }> = [];

  id() {
    return 'promptfoo:redteam:goat';
  }

  constructor(
    options: ProviderOptions & {
      maxTurns?: number;
      injectVar?: string;
      stateful?: boolean;
      excludeTargetOutputFromAgenticAttackGeneration?: boolean;
      continueAfterSuccess?: boolean;
    } = {},
  ) {
    if (neverGenerateRemote()) {
      throw new Error(`GOAT strategy requires remote grading to be enabled`);
    }
    invariant(typeof options.injectVar === 'string', 'Expected injectVar to be set');
    this.config = {
      maxTurns: options.maxTurns || 5,
      injectVar: options.injectVar,
      stateful: options.stateful ?? false,
      excludeTargetOutputFromAgenticAttackGeneration:
        options.excludeTargetOutputFromAgenticAttackGeneration ?? false,
      continueAfterSuccess: options.continueAfterSuccess ?? false,
    };
    logger.debug(
      `[GOAT] Constructor options: ${JSON.stringify({
        injectVar: options.injectVar,
        maxTurns: options.maxTurns,
        stateful: options.stateful,
        continueAfterSuccess: options.continueAfterSuccess,
      })}`,
    );
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<GoatResponse> {
    // Reset successful attacks array for each new call
    this.successfulAttacks = [];

    let response: Response | undefined = undefined;
    logger.debug(`[GOAT] callApi context: ${safeJsonStringify(context)}`);
    invariant(context?.originalProvider, 'Expected originalProvider to be set');
    invariant(context?.vars, 'Expected vars to be set');

    const targetProvider: ApiProvider | undefined = context?.originalProvider;
    invariant(targetProvider, 'Expected originalProvider to be set');

    const messages: Message[] = [];
    const totalTokenUsage = {
      total: 0,
      prompt: 0,
      completion: 0,
      numRequests: 0,
      cached: 0,
    };

    let lastTargetResponse: ProviderResponse | undefined = undefined;

    let assertToUse: Assertion | AssertionSet | undefined;
    let graderPassed: boolean | undefined;
    const { getGraderById } = await import('../graders');
    let test: AtomicTestCase | undefined;

    if (context?.test) {
      test = context?.test;
      assertToUse = test?.assert?.find(
        (a: { type: string }) => a.type && a.type.includes(test?.metadata?.pluginId),
      );

      // Fallback: if no assertion matches the pluginId, use the first assertion with a type
      if (!assertToUse) {
        assertToUse = test?.assert?.find((a: { type: string }) => a.type);
      }
    }

    let previousAttackerMessage = '';
    let previousTargetOutput = '';

    for (let turn = 0; turn < this.config.maxTurns; turn++) {
      try {
        // Handle unblocking logic BEFORE attack (skip on first turn)
        if (turn > 0 && previousTargetOutput) {
          const unblockingResult = await tryUnblocking({
            messages,
            lastResponse: previousTargetOutput,
            goal: context?.test?.metadata?.goal || context?.vars[this.config.injectVar],
            purpose: context?.test?.metadata?.purpose,
          });

          if (unblockingResult.tokenUsage) {
            totalTokenUsage.total += unblockingResult.tokenUsage.total || 0;
            totalTokenUsage.prompt += unblockingResult.tokenUsage.prompt || 0;
            totalTokenUsage.completion += unblockingResult.tokenUsage.completion || 0;
            totalTokenUsage.numRequests += unblockingResult.tokenUsage.numRequests ?? 1;
            totalTokenUsage.cached += unblockingResult.tokenUsage.cached || 0;
          }

          if (unblockingResult.success && unblockingResult.unblockingPrompt) {
            logger.debug(
              `[GOAT] Sending unblocking response: ${unblockingResult.unblockingPrompt}`,
            );

            messages.push({ role: 'user', content: unblockingResult.unblockingPrompt });

            const unblockingTargetPrompt = this.config.stateful
              ? unblockingResult.unblockingPrompt
              : JSON.stringify(messages);

            const unblockingResponse = await targetProvider.callApi(
              unblockingTargetPrompt,
              context,
              options,
            );

            if (!unblockingResponse.cached && targetProvider.delay && targetProvider.delay > 0) {
              logger.debug(`Sleeping for ${targetProvider.delay}ms`);
              await sleep(targetProvider.delay);
            }

            if (unblockingResponse.tokenUsage) {
              totalTokenUsage.total += unblockingResponse.tokenUsage.total || 0;
              totalTokenUsage.prompt += unblockingResponse.tokenUsage.prompt || 0;
              totalTokenUsage.completion += unblockingResponse.tokenUsage.completion || 0;
              totalTokenUsage.numRequests += unblockingResponse.tokenUsage.numRequests ?? 1;
              totalTokenUsage.cached += unblockingResponse.tokenUsage.cached || 0;
            } else {
              totalTokenUsage.numRequests += 1;
            }

            const unblockingOutput =
              typeof unblockingResponse.output === 'string'
                ? unblockingResponse.output
                : safeJsonStringify(unblockingResponse.output);

            if (unblockingOutput) {
              messages.push({ role: 'assistant', content: unblockingOutput });
            }

            if (unblockingResponse.error) {
              logger.error(`[GOAT] Target returned an error: ${unblockingResponse.error}`);
            }
          }
        }

        // Generate and send attack
        let body: string;
        let failureReason: string | undefined;
        if (this.config.excludeTargetOutputFromAgenticAttackGeneration && turn > 0) {
          body = JSON.stringify({
            goal: context?.test?.metadata?.goal || context?.vars[this.config.injectVar],
            targetOutput: previousTargetOutput,
            attackAttempt: previousAttackerMessage,
            task: 'extract-goat-failure',
          });
          logger.debug(`[GOAT] Sending request to ${getRemoteGenerationUrl()}: ${body}`);
          response = await fetch(getRemoteGenerationUrl(), {
            body,
            headers: {
              'Content-Type': 'application/json',
            },
            method: 'POST',
          });
          const data = (await response.json()) as ExtractAttackFailureResponse;

          if (!data.message) {
            logger.info(`[GOAT] Invalid message from GOAT, skipping turn: ${JSON.stringify(data)}`);
            continue;
          }
          failureReason = data.message;
          logger.debug(`[GOAT] Previous attack attempt failure reason: ${failureReason}`);
        }

        body = JSON.stringify({
          goal: context?.test?.metadata?.goal || context?.vars[this.config.injectVar],
          i: turn,
          messages: this.config.excludeTargetOutputFromAgenticAttackGeneration
            ? messages.filter((m) => m.role !== 'assistant')
            : messages,
          prompt: context?.prompt?.raw,
          task: 'goat',
          version: VERSION,
          email: getUserEmail(),
          excludeTargetOutputFromAgenticAttackGeneration:
            this.config.excludeTargetOutputFromAgenticAttackGeneration,
          failureReason,
          purpose: context?.test?.metadata?.purpose,
        });

        logger.debug(`[GOAT] Sending request to ${getRemoteGenerationUrl()}: ${body}`);
        response = await fetch(getRemoteGenerationUrl(), {
          body,
          headers: {
            'Content-Type': 'application/json',
          },
          method: 'POST',
        });
        const data = await response.json();
        if (typeof data?.message !== 'object' || !data.message?.content || !data.message?.role) {
          logger.info(`[GOAT] Invalid message from GOAT, skipping turn: ${JSON.stringify(data)}`);
          continue;
        }
        const attackerMessage = data.message;

        previousAttackerMessage = attackerMessage?.content;

        const targetVars = {
          ...context.vars,
          [this.config.injectVar]: attackerMessage.content,
        };

        const renderedAttackerPrompt = await renderPrompt(
          context.prompt,
          targetVars,
          context.filters,
          targetProvider,
        );

        messages.push({
          role: attackerMessage.role,
          content: renderedAttackerPrompt,
        });

        if (data.tokenUsage) {
          totalTokenUsage.total += data.tokenUsage.total || 0;
          totalTokenUsage.prompt += data.tokenUsage.prompt || 0;
          totalTokenUsage.completion += data.tokenUsage.completion || 0;
          totalTokenUsage.numRequests += data.tokenUsage.numRequests ?? 1;
          totalTokenUsage.cached += data.tokenUsage.cached || 0;
        }
        logger.debug(
          dedent`
          ${chalk.bold.green(`GOAT turn ${turn} history:`)}
          ${chalk.cyan(JSON.stringify(messages, null, 2))}
        `,
        );

        const targetPrompt = this.config.stateful
          ? messages[messages.length - 1].content
          : JSON.stringify(messages);
        logger.debug(`GOAT turn ${turn} target prompt: ${renderedAttackerPrompt}`);
        const targetResponse = await targetProvider.callApi(targetPrompt, context, options);

        if (!targetResponse.cached && targetProvider.delay && targetProvider.delay > 0) {
          logger.debug(`Sleeping for ${targetProvider.delay}ms`);
          await sleep(targetProvider.delay);
        }

        logger.debug(`GOAT turn ${turn} target response: ${safeJsonStringify(targetResponse)}`);

        if (targetResponse.sessionId) {
          context = context ?? { vars: {}, prompt: { raw: '', label: 'target' } };
          context.vars.sessionId = targetResponse.sessionId;
        }
        if (targetResponse.error) {
          throw new Error(`[GOAT] Target returned an error: ${targetResponse.error}`);
        }
        invariant(
          targetResponse.output,
          `[GOAT] Expected target response output to be set, but got: ${safeJsonStringify(targetResponse)}`,
        );

        const stringifiedOutput =
          typeof targetResponse.output === 'string'
            ? targetResponse.output
            : safeJsonStringify(targetResponse.output);
        const finalOutput = stringifiedOutput;
        const finalResponse = targetResponse;

        if (!stringifiedOutput) {
          logger.debug(
            `[GOAT] Target response output is not a string or JSON: ${safeJsonStringify(targetResponse)}`,
          );
          continue;
        }

        messages.push({
          role: 'assistant',
          content: stringifiedOutput,
        });

        // Store the attack response for potential unblocking in next turn
        previousTargetOutput = stringifiedOutput;

        if (targetResponse.tokenUsage) {
          totalTokenUsage.total += targetResponse.tokenUsage.total || 0;
          totalTokenUsage.prompt += targetResponse.tokenUsage.prompt || 0;
          totalTokenUsage.completion += targetResponse.tokenUsage.completion || 0;
          totalTokenUsage.numRequests += targetResponse.tokenUsage.numRequests ?? 1;
          totalTokenUsage.cached += targetResponse.tokenUsage.cached || 0;
        } else {
          totalTokenUsage.numRequests += 1;
        }

        lastTargetResponse = finalResponse;

        const grader = assertToUse ? getGraderById(assertToUse.type) : undefined;
        if (test && grader && finalOutput) {
          const { grade } = await grader.getResult(
            attackerMessage.content,
            finalOutput,
            test,
            targetProvider,
            assertToUse && 'value' in assertToUse ? assertToUse.value : undefined,
          );
          graderPassed = grade.pass;
          if (grade.tokensUsed) {
            totalTokenUsage.total += grade.tokensUsed.total || 0;
            totalTokenUsage.prompt += grade.tokensUsed.prompt || 0;
            totalTokenUsage.completion += grade.tokensUsed.completion || 0;
            totalTokenUsage.cached += grade.tokensUsed.cached || 0;
          } else {
            totalTokenUsage.numRequests = (totalTokenUsage.numRequests || 0) + 1;
          }
        }

        if (graderPassed === false) {
          // Record successful attack
          this.successfulAttacks.push({
            turn,
            prompt: attackerMessage.content,
            response: stringifiedOutput,
          });

          // Only break early if continueAfterSuccess is false
          if (this.config.continueAfterSuccess) {
            // Continue to next turn
          } else {
            break;
          }
        }
      } catch (err) {
        logger.error(`Error in GOAT turn ${turn}: ${err}`);
      }
    }

    return {
      output: getLastMessageContent(messages, 'assistant') || '',
      metadata: {
        redteamFinalPrompt: getLastMessageContent(messages, 'user') || '',
        messages: messages as Record<string, any>[],
        stopReason:
          this.successfulAttacks.length > 0 && !this.config.continueAfterSuccess
            ? 'Grader failed'
            : 'Max turns reached',
        redteamHistory: messagesToRedteamHistory(messages),
        successfulAttacks: this.successfulAttacks,
        totalSuccessfulAttacks: this.successfulAttacks.length,
      },
      tokenUsage: totalTokenUsage,
      guardrails: lastTargetResponse?.guardrails,
    };
  }
}
