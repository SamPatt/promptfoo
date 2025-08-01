import React, { useEffect, useRef, useState } from 'react';

import ErrorBoundary from '@app/components/ErrorBoundary';
import { useToast } from '@app/hooks/useToast';
import {
  type EvaluateTable,
  type EvaluateTableOutput,
  type EvaluateTableRow,
  type FilterMode,
} from '@app/pages/eval/components/types';
import { callApi } from '@app/utils/api';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CloseIcon from '@mui/icons-material/Close';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import Box from '@mui/material/Box';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import TextField from '@mui/material/TextField';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import { FILE_METADATA_KEY } from '@promptfoo/constants';
import invariant from '@promptfoo/util/invariant';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table';
import yaml from 'js-yaml';
import ReactMarkdown from 'react-markdown';
import { Link, useNavigate } from 'react-router-dom';
import remarkGfm from 'remark-gfm';
import { useDebounce } from 'use-debounce';
import CustomMetrics from './CustomMetrics';
import EvalOutputCell from './EvalOutputCell';
import EvalOutputPromptDialog from './EvalOutputPromptDialog';
import MarkdownErrorBoundary from './MarkdownErrorBoundary';
import { useResultsViewSettingsStore, useTableStore } from './store';
import TruncatedText from './TruncatedText';
import type { CellContext, ColumnDef, VisibilityState } from '@tanstack/table-core';

import type { TruncatedTextProps } from './TruncatedText';
import './ResultsTable.css';

import ButtonGroup from '@mui/material/ButtonGroup';

const VARIABLE_COLUMN_SIZE_PX = 200;
const PROMPT_COLUMN_SIZE_PX = 400;
const DESCRIPTION_COLUMN_SIZE_PX = 100;

function formatRowOutput(output: EvaluateTableOutput | string) {
  if (typeof output === 'string') {
    // Backwards compatibility for 0.15.0 breaking change. Remove eventually.
    const pass = output.startsWith('[PASS]');
    let text = output;
    if (output.startsWith('[PASS]')) {
      text = text.slice('[PASS]'.length);
    } else if (output.startsWith('[FAIL]')) {
      text = text.slice('[FAIL]'.length);
    } else if (output.startsWith('[ERROR]')) {
      text = text.slice('[ERROR]'.length);
    }
    return {
      text,
      pass,
      score: pass ? 1 : 0,
    };
  }
  return output;
}

function TableHeader({
  text,
  maxLength,
  expandedText,
  resourceId,
  className,
}: TruncatedTextProps & { expandedText?: string; resourceId?: string; className?: string }) {
  const [promptOpen, setPromptOpen] = React.useState(false);
  const handlePromptOpen = () => {
    setPromptOpen(true);
  };
  const handlePromptClose = () => {
    setPromptOpen(false);
  };
  return (
    <div className={`${className || ''}`}>
      <TruncatedText text={text} maxLength={maxLength} />
      {expandedText && (
        <>
          <Tooltip title="View prompt">
            <span className="action" onClick={handlePromptOpen}>
              🔎
            </span>
          </Tooltip>
          {promptOpen && (
            <EvalOutputPromptDialog
              open={promptOpen}
              onClose={handlePromptClose}
              prompt={expandedText}
            />
          )}
          {resourceId && (
            <Tooltip title="View other evals and datasets for this prompt">
              <span className="action">
                <Link to={`/prompts/?id=${resourceId}`} target="_blank">
                  <OpenInNewIcon fontSize="small" />
                </Link>
              </span>
            </Tooltip>
          )}
        </>
      )}
    </div>
  );
}

interface ResultsTableProps {
  maxTextLength: number;
  columnVisibility: VisibilityState;
  wordBreak: 'break-word' | 'break-all';
  filterMode: FilterMode;
  failureFilter: { [key: string]: boolean };
  searchText: string;
  debouncedSearchText?: string;
  showStats: boolean;
  onFailureFilterToggle: (columnId: string, checked: boolean) => void;
  onSearchTextChange: (text: string) => void;
  setFilterMode: (mode: FilterMode) => void;
  zoom: number;
}

interface ExtendedEvaluateTableOutput extends EvaluateTableOutput {
  originalRowIndex?: number;
  originalPromptIndex?: number;
}

interface ExtendedEvaluateTableRow extends EvaluateTableRow {
  outputs: ExtendedEvaluateTableOutput[];
}

function ResultsTable({
  maxTextLength,
  columnVisibility,
  wordBreak,
  filterMode,
  failureFilter,
  searchText,
  debouncedSearchText: externalDebouncedSearchText,
  showStats,
  onFailureFilterToggle,
  onSearchTextChange,
  setFilterMode,
  zoom,
}: ResultsTableProps) {
  const {
    evalId,
    table,
    setTable,
    config,
    version,
    filteredResultsCount,
    fetchEvalData,
    isFetching,
    filters,
    addFilter,
  } = useTableStore();
  const { inComparisonMode, comparisonEvalIds } = useResultsViewSettingsStore();

  const { showToast } = useToast();
  const navigate = useNavigate();

  invariant(table, 'Table should be defined');
  const { head, body } = table;

  const visiblePromptCount = React.useMemo(
    () => head.prompts.filter((_, idx) => columnVisibility[`Prompt ${idx + 1}`] !== false).length,
    [head.prompts, columnVisibility],
  );

  const [lightboxOpen, setLightboxOpen] = React.useState(false);
  const [lightboxImage, setLightboxImage] = React.useState<string | null>(null);
  const [pagination, setPagination] = React.useState<{ pageIndex: number; pageSize: number }>({
    pageIndex: 0,
    pageSize: filteredResultsCount > 10 ? 50 : 10,
  });

  /**
   * Reset the pagination state when the filtered results count changes.
   */
  React.useEffect(() => {
    setPagination({
      pageIndex: 0,
      pageSize: filteredResultsCount > 10 ? 50 : 10,
    });
  }, [filteredResultsCount]);

  const toggleLightbox = (url?: string) => {
    setLightboxImage(url || null);
    setLightboxOpen(!lightboxOpen);
  };

  const handleRating = React.useCallback(
    async (
      rowIndex: number,
      promptIndex: number,
      resultId: string,
      isPass?: boolean,
      score?: number,
      comment?: string,
    ) => {
      const updatedData = [...body];
      const updatedRow = { ...updatedData[rowIndex] };
      const updatedOutputs = [...updatedRow.outputs];
      const existingOutput = updatedOutputs[promptIndex];

      const finalPass = typeof isPass === 'undefined' ? existingOutput.pass : isPass;

      let finalScore = existingOutput.score;
      if (typeof score !== 'undefined') {
        finalScore = score;
      } else if (typeof isPass !== 'undefined') {
        finalScore = isPass ? 1 : 0;
      }

      updatedOutputs[promptIndex].pass = finalPass;
      updatedOutputs[promptIndex].score = finalScore;

      let componentResults = existingOutput.gradingResult?.componentResults;
      let modifiedComponentResults = false;

      if (typeof isPass !== 'undefined') {
        // Make a copy to avoid mutating the original
        componentResults = [...(componentResults || [])];
        modifiedComponentResults = true;

        const humanResultIndex = componentResults.findIndex(
          (result) => result.assertion?.type === 'human',
        );

        const newResult = {
          pass: finalPass,
          score: finalScore,
          reason: 'Manual result (overrides all other grading results)',
          comment,
          assertion: { type: 'human' as const },
        };

        if (humanResultIndex === -1) {
          componentResults.push(newResult);
        } else {
          componentResults[humanResultIndex] = newResult;
        }
      }

      // Build gradingResult, ensuring required fields are always present
      // Destructure to exclude componentResults initially
      const { componentResults: _, ...existingGradingResultWithoutComponents } =
        existingOutput.gradingResult || {};

      const gradingResult = {
        // Copy over existing fields except componentResults
        ...existingGradingResultWithoutComponents,
        // Ensure required fields have valid values
        pass: existingOutput.gradingResult?.pass ?? finalPass,
        score: existingOutput.gradingResult?.score ?? finalScore,
        reason: existingOutput.gradingResult?.reason ?? 'Manual result',
        // Always update comment
        comment,
      };

      // Only update pass/score/reason/assertion if we're actually rating (not just commenting)
      if (typeof isPass !== 'undefined' || typeof score !== 'undefined') {
        gradingResult.pass = finalPass;
        gradingResult.score = finalScore;
        gradingResult.reason = 'Manual result (overrides all other grading results)';
        gradingResult.assertion = existingOutput.gradingResult?.assertion || null;
      }

      // Only include componentResults if we modified them, or if we didn't modify them but they exist and are not empty
      if (modifiedComponentResults && componentResults) {
        (gradingResult as any).componentResults = componentResults;
      } else if (
        !modifiedComponentResults &&
        existingOutput.gradingResult?.componentResults &&
        existingOutput.gradingResult.componentResults.length > 0
      ) {
        (gradingResult as any).componentResults = existingOutput.gradingResult.componentResults;
      }

      updatedOutputs[promptIndex].gradingResult = gradingResult;
      updatedRow.outputs = updatedOutputs;
      updatedData[rowIndex] = updatedRow;
      const newTable: EvaluateTable = {
        head,
        body: updatedData,
      };

      setTable(newTable);
      if (inComparisonMode) {
        showToast('Ratings are not saved in comparison mode', 'warning');
      } else {
        try {
          let response;
          if (version && version >= 4) {
            response = await callApi(`/eval/${evalId}/results/${resultId}/rating`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ ...gradingResult }),
            });
          } else {
            response = await callApi(`/eval/${evalId}`, {
              method: 'PATCH',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ table: newTable }),
            });
          }
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
        } catch (error) {
          console.error('Failed to update table:', error);
        }
      }
    },
    [body, head, setTable, evalId, inComparisonMode, showToast],
  );

  const [localDebouncedSearchText] = useDebounce(searchText, 200);
  const debouncedSearchText = externalDebouncedSearchText || localDebouncedSearchText;
  const [isSearching, setIsSearching] = useState(false);

  React.useEffect(() => {
    setIsSearching(searchText !== debouncedSearchText && searchText !== '');
  }, [searchText, debouncedSearchText]);

  const tableBody = React.useMemo(() => {
    return body.map((row, rowIndex) => ({
      ...row,
      outputs: row.outputs.map((output, promptIndex) => ({
        ...output,
        originalRowIndex: rowIndex,
        originalPromptIndex: promptIndex,
      })),
    })) as ExtendedEvaluateTableRow[];
  }, [body]);

  const parseQueryParams = (queryString: string) => {
    return Object.fromEntries(new URLSearchParams(queryString));
  };

  React.useEffect(() => {
    setPagination({ ...pagination, pageIndex: 0 });
  }, [failureFilter, filterMode, debouncedSearchText, filters.appliedCount]);

  // Add a ref to track the current evalId to compare with new values
  const previousEvalIdRef = useRef<string | null>(null);

  // Reset pagination when evalId changes
  React.useEffect(() => {
    if (evalId !== previousEvalIdRef.current) {
      setPagination({ pageIndex: 0, pageSize: pagination.pageSize });
      previousEvalIdRef.current = evalId;

      // Don't fetch here - the parent component (Eval.tsx) is responsible
      // for the initial data load when changing evalId
    }
  }, [evalId]);

  // Only fetch data when pagination or filters change, but not when evalId changes
  React.useEffect(() => {
    if (!evalId) {
      return;
    }

    // Skip fetching if this is the first render for a new evalId
    // Data should already be loaded by Eval.tsx
    if (pagination.pageIndex === 0 && evalId !== previousEvalIdRef.current) {
      previousEvalIdRef.current = evalId;
      return;
    }

    console.log('Fetching data for filtering/pagination', evalId);

    fetchEvalData(evalId, {
      pageIndex: pagination.pageIndex,
      pageSize: pagination.pageSize,
      filterMode,
      searchText: debouncedSearchText,
      // Only pass the filters that have been applied (have a value).
      // For metadata filters, both field and value must be present.
      filters: Object.values(filters.values).filter((filter) =>
        filter.type === 'metadata' ? Boolean(filter.value && filter.field) : Boolean(filter.value),
      ),
      skipSettingEvalId: true, // Don't change evalId when paginating or filtering
    });
  }, [
    // evalId is NOT in the dependency array
    pagination.pageIndex,
    pagination.pageSize,
    filterMode,
    comparisonEvalIds,
    debouncedSearchText,
    fetchEvalData,
    filters.values,
    // Ensure this re-triggers for filter CxUD operations.
    filters.appliedCount,
  ]);

  // TODO(ian): Switch this to use prompt.metrics field once most clients have updated.
  const numGoodTests = React.useMemo(
    () => head.prompts.map((prompt) => prompt.metrics?.testPassCount || 0),
    [head.prompts, body],
  );

  const numTests = React.useMemo(
    () =>
      head.prompts.map(
        (prompt) => (prompt.metrics?.testPassCount ?? 0) + (prompt.metrics?.testFailCount ?? 0),
      ),
    [head.prompts, body],
  );

  const numAsserts = React.useMemo(
    () =>
      head.prompts.map(
        (prompt) => (prompt.metrics?.assertFailCount ?? 0) + (prompt.metrics?.assertPassCount ?? 0),
      ),
    [head.prompts, body],
  );

  const numGoodAsserts = React.useMemo(
    () => head.prompts.map((prompt) => prompt.metrics?.assertPassCount || 0),
    [head.prompts, body],
  );

  const columnHelper = React.useMemo(() => createColumnHelper<EvaluateTableRow>(), []);

  const { renderMarkdown } = useResultsViewSettingsStore();
  const variableColumns = React.useMemo(() => {
    if (head.vars.length > 0) {
      return [
        columnHelper.group({
          id: 'vars',
          header: () => <span className="font-bold">Variables</span>,
          columns: head.vars.map((varName, idx) =>
            columnHelper.accessor((row: EvaluateTableRow) => row.vars[idx], {
              id: `Variable ${idx + 1}`,
              header: () => (
                <TableHeader text={varName} maxLength={maxTextLength} className="font-bold" />
              ),
              cell: (info: CellContext<EvaluateTableRow, string>) => {
                let value: string | object = info.getValue();

                // Get the first output that has metadata for checking file metadata
                const row = info.row.original;
                const output = row.outputs && row.outputs.length > 0 ? row.outputs[0] : null;

                const fileMetadata = output?.metadata?.[FILE_METADATA_KEY] as
                  | Record<string, { path: string; type: string; format?: string }>
                  | undefined;
                const isMediaFile = fileMetadata && fileMetadata[varName];

                if (isMediaFile) {
                  // Handle various media types
                  const mediaMetadata = fileMetadata[varName];
                  const mediaType = mediaMetadata.type;
                  const format = mediaMetadata.format || '';
                  const mediaDataUrl = value.startsWith('data:')
                    ? value
                    : `data:${mediaType}/${format};base64,${value}`;

                  let mediaElement = null;

                  if (mediaType === 'audio') {
                    mediaElement = (
                      <audio controls style={{ maxWidth: '100%' }}>
                        <source src={mediaDataUrl} type={`audio/${format}`} />
                        Your browser does not support the audio element.
                      </audio>
                    );
                  } else if (mediaType === 'video') {
                    mediaElement = (
                      <video controls style={{ maxWidth: '100%', maxHeight: '200px' }}>
                        <source src={mediaDataUrl} type={`video/${format}`} />
                        Your browser does not support the video element.
                      </video>
                    );
                  } else if (mediaType === 'image') {
                    mediaElement = (
                      <img
                        src={mediaDataUrl}
                        alt="Input image"
                        style={{ maxWidth: '100%', maxHeight: '200px' }}
                        onClick={() => toggleLightbox?.(mediaDataUrl)}
                      />
                    );
                  }

                  if (mediaElement) {
                    return (
                      <div className="cell">
                        <div style={{ marginBottom: '8px' }}>{mediaElement}</div>
                        <Tooltip title="Original file path">
                          <span style={{ fontSize: '0.8em', color: '#666' }}>
                            {mediaMetadata.path} ({mediaType}/{format})
                          </span>
                        </Tooltip>
                      </div>
                    );
                  }
                }

                if (typeof value === 'object') {
                  value = JSON.stringify(value, null, 2);
                  if (renderMarkdown) {
                    value = `\`\`\`json\n${value}\n\`\`\``;
                  }
                }
                const truncatedValue =
                  value.length > maxTextLength
                    ? `${value.substring(0, maxTextLength - 3).trim()}...`
                    : value;
                return (
                  <div className="cell" data-capture="true">
                    {renderMarkdown ? (
                      <MarkdownErrorBoundary fallback={value}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{truncatedValue}</ReactMarkdown>
                      </MarkdownErrorBoundary>
                    ) : (
                      <TruncatedText text={value} maxLength={maxTextLength} />
                    )}
                  </div>
                );
              },
              size: VARIABLE_COLUMN_SIZE_PX,
            }),
          ),
        }),
      ];
    }
    return [];
  }, [columnHelper, head.vars, maxTextLength, renderMarkdown]);

  const getOutput = React.useCallback(
    (rowIndex: number, promptIndex: number) => {
      return tableBody[rowIndex].outputs[promptIndex];
    },
    [tableBody],
  );

  const getFirstOutput = React.useCallback(
    (rowIndex: number) => {
      return tableBody[rowIndex].outputs[0];
    },
    [tableBody],
  );

  const metricTotals = React.useMemo(() => {
    // Use the backend's already-correct namedScoresCount instead of recalculating
    const firstProvider = table?.head?.prompts?.[0];
    const backendCounts = firstProvider?.metrics?.namedScoresCount;

    if (backendCounts) {
      return backendCounts;
    }

    const totals: Record<string, number> = {};
    table?.body.forEach((row) => {
      row.test.assert?.forEach((assertion) => {
        if (assertion.metric) {
          totals[assertion.metric] = (totals[assertion.metric] || 0) + 1;
        }
        if ('assert' in assertion && Array.isArray(assertion.assert)) {
          assertion.assert.forEach((subAssertion) => {
            if ('metric' in subAssertion && subAssertion.metric) {
              totals[subAssertion.metric] = (totals[subAssertion.metric] || 0) + 1;
            }
          });
        }
      });
    });
    return totals;
  }, [table?.head?.prompts, table?.body]);

  const handleMetricFilterClick = React.useCallback(
    (metric: string | null) => {
      if (!metric) {
        return;
      }

      const filter = {
        type: 'metric' as const,
        operator: 'equals' as const,
        value: metric,
        logicOperator: 'or' as const,
      };

      // If this filter is already applied, do not re-apply it.
      if (
        Object.values(filters.values).find(
          (f) =>
            f.type === filter.type &&
            f.value === filter.value &&
            f.operator === filter.operator &&
            f.logicOperator === filter.logicOperator,
        )
      ) {
        return;
      }

      addFilter(filter);
    },
    [addFilter, filters.values],
  );

  const promptColumns = React.useMemo(() => {
    return [
      columnHelper.group({
        id: 'prompts',
        header: () => <span className="font-bold">Outputs</span>,
        columns: head.prompts.map((prompt, idx) =>
          columnHelper.accessor((row: EvaluateTableRow) => formatRowOutput(row.outputs[idx]), {
            id: `Prompt ${idx + 1}`,
            header: () => {
              const pct =
                numGoodTests[idx] && numTests[idx]
                  ? ((numGoodTests[idx] / numTests[idx]) * 100.0).toFixed(2)
                  : '0.00';
              const columnId = `Prompt ${idx + 1}`;
              const isChecked = failureFilter[columnId] || false;

              const details = showStats ? (
                <div className="prompt-detail collapse-hidden">
                  {numAsserts[idx] ? (
                    <div>
                      <strong>Asserts:</strong> {numGoodAsserts[idx]}/{numAsserts[idx]} passed
                    </div>
                  ) : null}
                  {prompt.metrics?.cost ? (
                    <div>
                      <strong>Total Cost:</strong>{' '}
                      <Tooltip
                        title={`Average: $${Intl.NumberFormat(undefined, {
                          minimumFractionDigits: 1,
                          maximumFractionDigits: prompt.metrics.cost / numTests[idx] >= 1 ? 2 : 4,
                        }).format(prompt.metrics.cost / numTests[idx])} per test`}
                      >
                        <span style={{ cursor: 'help' }}>
                          $
                          {Intl.NumberFormat(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: prompt.metrics.cost >= 1 ? 2 : 4,
                          }).format(prompt.metrics.cost)}
                        </span>
                      </Tooltip>
                    </div>
                  ) : null}
                  {prompt.metrics?.tokenUsage?.total ? (
                    <div>
                      <strong>Total Tokens:</strong>{' '}
                      {Intl.NumberFormat(undefined, {
                        maximumFractionDigits: 0,
                      }).format(prompt.metrics.tokenUsage.total)}
                    </div>
                  ) : null}

                  {prompt.metrics?.tokenUsage?.total ? (
                    <div>
                      <strong>Avg Tokens:</strong>{' '}
                      {Intl.NumberFormat(undefined, {
                        maximumFractionDigits: 0,
                      }).format(prompt.metrics.tokenUsage.total / numTests[idx])}
                    </div>
                  ) : null}
                  {prompt.metrics?.totalLatencyMs ? (
                    <div>
                      <strong>Avg Latency:</strong>{' '}
                      {Intl.NumberFormat(undefined, {
                        maximumFractionDigits: 0,
                      }).format(prompt.metrics.totalLatencyMs / numTests[idx])}{' '}
                      ms
                    </div>
                  ) : null}
                  {prompt.metrics?.totalLatencyMs && prompt.metrics?.tokenUsage?.completion ? (
                    <div>
                      <strong>Tokens/Sec:</strong>{' '}
                      {prompt.metrics.totalLatencyMs > 0
                        ? Intl.NumberFormat(undefined, {
                            maximumFractionDigits: 0,
                          }).format(
                            prompt.metrics.tokenUsage.completion /
                              (prompt.metrics.totalLatencyMs / 1000),
                          )
                        : '0'}
                    </div>
                  ) : null}
                </div>
              ) : null;
              const providerConfig = Array.isArray(config?.providers)
                ? config.providers[idx]
                : undefined;

              let providerParts: string[] = [];
              try {
                if (prompt.provider && typeof prompt.provider === 'string') {
                  providerParts = prompt.provider.split(':');
                } else if (prompt.provider) {
                  const providerObj = prompt.provider as any; // Use any for flexible typing
                  const providerId =
                    typeof providerObj === 'object' && providerObj !== null
                      ? providerObj.id || JSON.stringify(providerObj)
                      : String(providerObj);
                  providerParts = [providerId];
                }
              } catch (error) {
                console.error('Error parsing provider:', error);
                providerParts = ['Error parsing provider'];
              }

              const providerDisplay = (
                <Tooltip title={providerConfig ? <pre>{yaml.dump(providerConfig)}</pre> : ''}>
                  {providerParts.length > 1 ? (
                    <>
                      {providerParts[0]}:<strong>{providerParts.slice(1).join(':')}</strong>
                    </>
                  ) : (
                    <strong>{providerParts[0] || 'Unknown provider'}</strong>
                  )}
                </Tooltip>
              );
              return (
                <div className="output-header">
                  <div className="pills collapse-font-small">
                    {prompt.provider ? <div className="provider">{providerDisplay}</div> : null}
                    <div className="summary">
                      <div
                        className={`highlight ${
                          numGoodTests[idx] && numTests[idx]
                            ? `success-${Math.round(((numGoodTests[idx] / numTests[idx]) * 100) / 20) * 20}`
                            : 'success-0'
                        }`}
                      >
                        <strong>{pct}% passing</strong> ({numGoodTests[idx]}/{numTests[idx]} cases)
                      </div>
                    </div>
                    {prompt.metrics?.testErrorCount && prompt.metrics.testErrorCount > 0 ? (
                      <div className="summary error-pill" onClick={() => setFilterMode('errors')}>
                        <div className="highlight fail">
                          <strong>Errors:</strong> {prompt.metrics?.testErrorCount || 0}
                        </div>
                      </div>
                    ) : null}
                    {prompt.metrics?.namedScores &&
                    Object.keys(prompt.metrics.namedScores).length > 0 ? (
                      <Box className="collapse-hidden">
                        <CustomMetrics
                          lookup={prompt.metrics.namedScores}
                          counts={prompt.metrics.namedScoresCount}
                          metricTotals={metricTotals}
                          onSearchTextChange={onSearchTextChange}
                          onMetricFilter={handleMetricFilterClick}
                        />
                      </Box>
                    ) : null}
                    {/* TODO(ian): Remove backwards compatibility for prompt.provider added 12/26/23 */}
                  </div>
                  <TableHeader
                    className="prompt-container collapse-font-small"
                    text={prompt.label || prompt.display || prompt.raw}
                    expandedText={prompt.raw}
                    maxLength={maxTextLength}
                    resourceId={prompt.id}
                  />
                  {details}
                  {filterMode === 'failures' && head.prompts.length > 1 && (
                    <FormControlLabel
                      sx={{
                        '& .MuiFormControlLabel-label': {
                          fontSize: '0.75rem',
                        },
                      }}
                      control={
                        <Checkbox
                          checked={isChecked}
                          onChange={(event) =>
                            onFailureFilterToggle(columnId, event.target.checked)
                          }
                        />
                      }
                      label="Show failures"
                    />
                  )}
                </div>
              );
            },
            cell: (info: CellContext<EvaluateTableRow, EvaluateTableOutput>) => {
              const output = getOutput(info.row.index, idx);
              return output ? (
                <ErrorBoundary
                  name={`EvalOutputCell-${info.row.index}-${idx}`}
                  fallback={
                    <div style={{ padding: '20px', color: 'red', fontSize: '12px' }}>
                      Error loading cell
                    </div>
                  }
                >
                  <EvalOutputCell
                    output={output}
                    maxTextLength={maxTextLength}
                    rowIndex={info.row.index}
                    promptIndex={idx}
                    onRating={handleRating.bind(
                      null,
                      output.originalRowIndex ?? info.row.index,
                      output.originalPromptIndex ?? idx,
                      output.id,
                    )}
                    firstOutput={getFirstOutput(info.row.index)}
                    showDiffs={filterMode === 'different' && visiblePromptCount > 1}
                    searchText={debouncedSearchText}
                    showStats={showStats}
                    evaluationId={evalId || undefined}
                    testCaseId={info.row.original.test?.metadata?.testCaseId || output.id}
                    onMetricFilter={handleMetricFilterClick}
                  />
                </ErrorBoundary>
              ) : (
                <div style={{ padding: '20px' }}>'Test still in progress...'</div>
              );
            },
            size: PROMPT_COLUMN_SIZE_PX,
          }),
        ),
      }),
    ];
  }, [
    body.length,
    config?.providers,
    columnHelper,
    failureFilter,
    filterMode,
    getFirstOutput,
    getOutput,
    handleRating,
    head.prompts,
    maxTextLength,
    metricTotals,
    numAsserts,
    numGoodAsserts,
    numGoodTests,
    onFailureFilterToggle,
    debouncedSearchText,
    showStats,
    filters.appliedCount,
    handleMetricFilterClick,
  ]);

  const descriptionColumn = React.useMemo(() => {
    const hasAnyDescriptions = body.some((row) => row.test.description);
    if (hasAnyDescriptions) {
      return {
        accessorFn: (row: EvaluateTableRow) => row.test.description || '',
        id: 'description',
        header: () => <span className="font-bold">Description</span>,
        cell: (info: CellContext<EvaluateTableRow, unknown>) => (
          <div className="cell">
            <TruncatedText text={String(info.getValue())} maxLength={maxTextLength} />
          </div>
        ),
        size: DESCRIPTION_COLUMN_SIZE_PX,
      };
    }
    return null;
  }, [body, maxTextLength]);

  const columns = React.useMemo(() => {
    const cols: ColumnDef<EvaluateTableRow, unknown>[] = [];
    if (descriptionColumn) {
      cols.push(descriptionColumn);
    }
    cols.push(...variableColumns, ...promptColumns);
    return cols;
  }, [descriptionColumn, variableColumns, promptColumns]);

  const pageCount = Math.ceil(filteredResultsCount / pagination.pageSize);
  const reactTable = useReactTable({
    data: tableBody,
    columns,
    columnResizeMode: 'onChange',
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    pageCount,
    state: {
      columnVisibility,
      pagination,
    },
    enableColumnResizing: true,
  });

  const { stickyHeader, setStickyHeader } = useResultsViewSettingsStore();

  const clearRowIdFromUrl = React.useCallback(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.has('rowId')) {
      url.searchParams.delete('rowId');
      navigate({ pathname: url.pathname, search: url.search }, { replace: true });
    }
  }, [navigate]);

  useEffect(() => {
    const params = parseQueryParams(window.location.search);
    const rowId = params['rowId'];

    if (rowId && Number.isInteger(Number(rowId))) {
      const parsedRowId = Number(rowId);
      const rowIndex = Math.max(0, Math.min(parsedRowId - 1, filteredResultsCount - 1));

      let hasScrolled = false;

      const rowPageIndex = Math.floor(rowIndex / pagination.pageSize);

      const maxPageIndex = pageCount - 1;
      const safeRowPageIndex = Math.min(rowPageIndex, maxPageIndex);

      if (pagination.pageIndex !== safeRowPageIndex) {
        setPagination((prev) => ({
          ...prev,
          pageIndex: safeRowPageIndex,
        }));
      }

      const scrollToRow = () => {
        if (hasScrolled) {
          return;
        }

        const localRowIndex = rowIndex % pagination.pageSize;
        const rowElement = document.querySelector(`#row-${localRowIndex}`);

        if (rowElement) {
          hasScrolled = true;

          requestAnimationFrame(() => {
            rowElement.scrollIntoView({
              behavior: 'smooth',
              block: 'nearest',
            });
          });
        }
      };

      const tableContainer =
        document.querySelector('.results-table')?.parentElement || document.body;

      const observer = new MutationObserver(() => {
        if (hasScrolled) {
          observer.disconnect();
          return;
        }

        const localRowIndex = rowIndex % pagination.pageSize;
        const rowElement = document.querySelector(`#row-${localRowIndex}`);
        if (rowElement) {
          scrollToRow();
          observer.disconnect();
        }
      });

      observer.observe(tableContainer, {
        childList: true,
        subtree: true,
      });

      const timeoutId = setTimeout(() => {
        if (!hasScrolled) {
          console.warn('Timeout reached while waiting for row to be rendered');
          observer.disconnect();
        }
      }, 5000);

      return () => {
        observer.disconnect();
        clearTimeout(timeoutId);
      };
    }
  }, [pagination.pageIndex, pagination.pageSize, reactTable, filteredResultsCount]);

  const tableWidth = React.useMemo(() => {
    let width = 0;
    if (descriptionColumn) {
      width += DESCRIPTION_COLUMN_SIZE_PX;
    }
    width += head.vars.length * VARIABLE_COLUMN_SIZE_PX;
    width += head.prompts.length * PROMPT_COLUMN_SIZE_PX;
    return width;
  }, [descriptionColumn, head.vars.length, head.prompts.length]);

  return (
    // NOTE: It's important that the JSX Fragment is the top-level element within the DOM tree
    // of this component. This ensures that the pagination footer is always pinned to the bottom
    // of the viewport (because the parent container is a flexbox).
    <>
      {isSearching && searchText && (
        <Box
          sx={{
            position: 'fixed',
            top: '60px',
            right: '20px',
            zIndex: 1000,
            backgroundColor: 'rgba(25, 118, 210, 0.1)',
            padding: '5px 10px',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <Typography variant="body2" color="primary">
            Searching...
          </Typography>
        </Box>
      )}

      {filteredResultsCount === 0 &&
        !isFetching &&
        (debouncedSearchText || filterMode !== 'all' || filters.appliedCount > 0) && (
          <Box
            sx={{
              padding: '20px',
              textAlign: 'center',
              backgroundColor: 'rgba(0, 0, 0, 0.03)',
              borderRadius: '4px',
              marginTop: '20px',
              marginBottom: '20px',
            }}
          >
            <Typography variant="body1">No results found for the current filters.</Typography>
          </Box>
        )}

      <div
        id="results-table-container"
        style={{
          zoom,
          borderTop: '1px solid',
          borderColor: stickyHeader ? 'transparent' : 'var(--border-color)',
          // Grow vertically into any empty space; this applies when total number of evals is so few that the table otherwise
          // won't extend to the bottom of the viewport.
          flexGrow: 1,
        }}
      >
        <table
          className={`results-table firefox-fix ${maxTextLength <= 25 ? 'compact' : ''}`}
          style={{
            wordBreak,
            width: `${tableWidth}px`,
          }}
        >
          <thead className={`${stickyHeader && 'sticky'}`}>
            {stickyHeader && (
              // TODO: Fix this position in the CSS.
              <div className="header-dismiss">
                <IconButton
                  onClick={() => setStickyHeader(false)}
                  size="small"
                  sx={{ color: 'text.primary' }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </div>
            )}
            {reactTable.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="header">
                {headerGroup.headers.map((header) => {
                  const isMetadataCol =
                    header.column.id.startsWith('Variable') || header.column.id === 'description';
                  const isFinalRow = headerGroup.depth === 1;

                  return (
                    <th
                      key={header.id}
                      colSpan={header.colSpan}
                      style={{
                        width: header.getSize(),
                        borderBottom: !isMetadataCol && isFinalRow ? '2px solid #888' : 'none',
                        height: isFinalRow ? 'fit-content' : 'auto',
                      }}
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(header.column.columnDef.header, header.getContext())}
                      <div
                        onMouseDown={header.getResizeHandler()}
                        onTouchStart={header.getResizeHandler()}
                        className={`resizer ${header.column.getIsResizing() ? 'isResizing' : ''}`}
                      />
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {reactTable.getRowModel().rows.map((row, rowIndex) => {
              let colBorderDrawn = false;

              return (
                <tr key={row.id} id={`row-${row.index % pagination.pageSize}`}>
                  {row.getVisibleCells().map((cell) => {
                    const isMetadataCol =
                      cell.column.id.startsWith('Variable') || cell.column.id === 'description';
                    const shouldDrawColBorder = !isMetadataCol && !colBorderDrawn;
                    if (shouldDrawColBorder) {
                      colBorderDrawn = true;
                    }

                    let cellContent = flexRender(cell.column.columnDef.cell, cell.getContext());
                    const value = cell.getValue();
                    if (
                      typeof value === 'string' &&
                      (value.match(/^data:(image\/[a-z]+|application\/octet-stream);base64,/) ||
                        value.match(/^[A-Za-z0-9+/]{20,}={0,2}$/))
                    ) {
                      const imgSrc = value.startsWith('data:')
                        ? value
                        : `data:image/jpeg;base64,${value}`;
                      cellContent = (
                        <>
                          <img
                            src={imgSrc}
                            alt="Base64 encoded image"
                            style={{
                              maxWidth: '100%',
                              height: 'auto',
                              cursor: 'pointer',
                            }}
                            onClick={() => toggleLightbox(imgSrc)}
                          />
                          {lightboxOpen && lightboxImage === imgSrc && (
                            <div className="lightbox" onClick={() => toggleLightbox()}>
                              <img
                                src={lightboxImage}
                                alt="Lightbox"
                                style={{
                                  maxWidth: '90%',
                                  maxHeight: '90vh',
                                  objectFit: 'contain',
                                }}
                              />
                            </div>
                          )}
                        </>
                      );
                    }

                    return (
                      <td
                        key={cell.id}
                        style={{
                          width: cell.column.getSize(),
                        }}
                        className={`${isMetadataCol ? 'variable' : ''}${shouldDrawColBorder ? 'first-prompt-col' : 'second-prompt-column'}`}
                      >
                        {cellContent}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <Box
        className="pagination"
        px={2}
        mx={-2}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          backgroundColor: 'background.paper',
          borderTop: '1px solid',
          borderColor: 'divider',
          width: '100vw',
          boxShadow: 3,
        }}
      >
        <Box>
          Showing{' '}
          {filteredResultsCount === 0 ? (
            <Typography component="span" sx={{ fontWeight: 600 }}>
              0
            </Typography>
          ) : (
            <>
              <Typography component="span" sx={{ fontWeight: 600 }}>
                {pagination.pageIndex * pagination.pageSize + 1}
              </Typography>{' '}
              to{' '}
              <Typography component="span" sx={{ fontWeight: 600 }}>
                {Math.min((pagination.pageIndex + 1) * pagination.pageSize, filteredResultsCount)}
              </Typography>{' '}
              of{' '}
              <Typography component="span" sx={{ fontWeight: 600 }}>
                {filteredResultsCount}
              </Typography>
            </>
          )}{' '}
          results
        </Box>

        {filteredResultsCount > 0 && (
          <Box>
            Page{' '}
            <Typography component="span" sx={{ fontWeight: 600 }}>
              {reactTable.getState().pagination.pageIndex + 1}
            </Typography>{' '}
            of{' '}
            <Typography component="span" sx={{ fontWeight: 600 }}>
              {pageCount}
            </Typography>
          </Box>
        )}

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* PAGE SIZE SELECTOR */}
          <Typography component="span" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <span>Results per page:</span>
            <Select
              value={pagination.pageSize}
              onChange={(e) => {
                setPagination((prev) => ({
                  ...prev,
                  pageSize: Number(e.target.value),
                }));
                window.scrollTo(0, 0);
              }}
              displayEmpty
              inputProps={{ 'aria-label': 'Results per page' }}
              size="small"
              sx={{ m: 1, minWidth: 80 }}
              disabled={filteredResultsCount <= 10}
            >
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={50} disabled={filteredResultsCount <= 10}>
                50
              </MenuItem>
              <MenuItem value={100} disabled={filteredResultsCount <= 50}>
                100
              </MenuItem>
              <MenuItem value={500} disabled={filteredResultsCount <= 100}>
                500
              </MenuItem>
              <MenuItem value={1000} disabled={filteredResultsCount <= 500}>
                1000
              </MenuItem>
            </Select>
          </Typography>

          {/* PAGE NAVIGATOR */}
          <Typography
            component="span"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              opacity: pageCount > 1 ? 1 : 0.5,
              pointerEvents: pageCount > 1 ? 'auto' : 'none',
            }}
          >
            <span>Go to:</span>
            <TextField
              size="small"
              type="number"
              onChange={(e) => {
                const page = e.target.value ? Number(e.target.value) - 1 : null;
                if (page !== null && page >= 0 && page < pageCount) {
                  setPagination((prev) => ({
                    ...prev,
                    pageIndex: Math.min(Math.max(page, 0), pageCount - 1),
                  }));
                  clearRowIdFromUrl();
                }
              }}
              sx={{
                width: '60px',
                textAlign: 'center',
              }}
              slotProps={{
                htmlInput: {
                  min: 1,
                  max: pageCount,
                },
              }}
              disabled={pageCount === 1}
            />
          </Typography>

          {/* PAGE NAVIGATION BUTTONS */}
          <ButtonGroup>
            <IconButton
              onClick={() => {
                setPagination((prev) => ({
                  ...prev,
                  pageIndex: Math.max(prev.pageIndex - 1, 0),
                }));
                clearRowIdFromUrl();
                window.scrollTo(0, 0);
              }}
              disabled={reactTable.getState().pagination.pageIndex === 0}
            >
              <ArrowBackIcon />
            </IconButton>
            <IconButton
              onClick={() => {
                setPagination((prev) => ({
                  ...prev,
                  pageIndex: Math.min(prev.pageIndex + 1, pageCount - 1),
                }));
                clearRowIdFromUrl();
                window.scrollTo(0, 0);
              }}
              disabled={reactTable.getState().pagination.pageIndex + 1 >= pageCount}
            >
              <ArrowForwardIcon />
            </IconButton>
          </ButtonGroup>
        </Box>
      </Box>
    </>
  );
}

export default React.memo(ResultsTable);
