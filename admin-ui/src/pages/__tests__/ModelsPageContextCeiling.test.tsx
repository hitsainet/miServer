/**
 * The Models page wires the details modal's "Load once with this ceiling" to the load (2026-11-21).
 *
 * MUTATION CONTROL (U5): render ModelDetailsModal without onLoadWithCeiling -> the button is not
 * offered and this fails; pass `load(id)` instead -> the ceiling is lost and this fails.
 */

import { describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ModelInfo } from '@/types';

const load = vi.fn();
const MODEL: ModelInfo = {
  id: 7, name: 'Qwen3.6-27B-GGUF:Q4_K_M', repo_id: 'unsloth/Qwen3.6-27B-GGUF',
  source: 'huggingface', quantization: 'Q4', params: '27B', disk_size_mb: 16038,
  estimated_memory_mb: null, local_path: '', status: 'ready', created_at: '2026-11-21T00:00:00Z',
  updated_at: '2026-11-21T00:00:00Z', gguf_label: 'Q4_K_M', context_ceiling: null,
};

vi.mock('@hooks/useModels', () => ({
  useModels: () => ({
    models: [MODEL], isLoading: false, downloadModel: vi.fn(), isDownloading: false, load,
    isLoadingModel: false, unloadModel: vi.fn(), isUnloading: false, delete: vi.fn(),
    isDeleting: false, previewModel: vi.fn(), isPreviewingModel: false, previewData: null,
    clearPreview: vi.fn(), lockModel: vi.fn(), unlockModel: vi.fn(), isLocking: false,
    isUnlockingModel: false,
  }),
}));
vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return {
    ...real,
    modelApi: {
      ...real.modelApi,
      probeCapacity: vi.fn(() => new Promise(() => {})),
      probeLimitRecommendation: vi.fn(() => new Promise(() => {})),
    },
  };
});

const { ModelsPage } = await import('../ModelsPage');

describe('ModelsPage context ceiling', () => {
  it('the page wires the load once to a load carrying the ceiling', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><ModelsPage /></QueryClientProvider>);
    await userEvent.click(screen.getByText('Qwen3.6-27B-GGUF:Q4_K_M'));
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'max');
    await userEvent.click(screen.getByTestId('context-ceiling-load-once'));
    expect(load).toHaveBeenCalledWith({ id: 7, contextCeiling: 'max' });
  });
});
