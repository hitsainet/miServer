/**
 * The GGUF context ceiling control (2026-11-21): it shows the ceiling and what the loaded model
 * got, sends `PUT /api/models/{id}/context-ceiling` with the right body, and loads once with a
 * ceiling of its own.
 *
 * MUTATION CONTROLS (recorded in 0xcc/reviews/gguf_context_ceiling_2026-11-21.md):
 *   U1 setContextCeiling sends no body                       -> the modelApi body tests
 *   U2 Save sends the stored value instead of the chosen one -> "sends max / tokens / null"
 *   U3 modelApi.load drops context_ceiling                   -> "a load once sends its ceiling"
 *   U4 the modal does not render the panel                   -> "the details modal shows it"
 *   U5 the page does not pass onLoadWithCeiling              -> "the page wires the load once"
 *   U6 the hook drops the ceiling it was given               -> "the hook passes it to the API"
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import type { ModelInfo } from '@/types';

const api = { setContextCeiling: vi.fn(), load: vi.fn() };
vi.mock('@/services/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('@/services/api')>();
  return { ...real, modelApi: { ...real.modelApi, ...api } };
});
const { ContextCeilingPanel } = await import('../ContextCeilingPanel');
const { ModelDetailsModal } = await import('../ModelDetailsModal');
const { useModels } = await import('@/hooks/useModels');

const GGUF: ModelInfo = {
  id: 7, name: 'Qwen3.6-27B-GGUF:Q4_K_M', repo_id: 'unsloth/Qwen3.6-27B-GGUF',
  source: 'huggingface', quantization: 'Q4', params: '27B', disk_size_mb: 16038,
  estimated_memory_mb: null, local_path: '', status: 'ready', created_at: '2026-11-21T00:00:00Z',
  updated_at: '2026-11-21T00:00:00Z', gguf_label: 'Q4_K_M', context_ceiling: null,
};

const LOADED: ModelInfo = {
  ...GGUF, status: 'loaded', context_ceiling: 0, context_length: 155648, kv_cache_type: 'q8_0',
  supports_embeddings: false,
  gguf_load: {
    kv_cache_type: 'q8_0', kv_cache_reason: 'q8_0: GGUF_KV_CACHE_TYPE', embeddings: false,
    embeddings_reason: 'off: the recurrent state does not fit',
    context: { ceiling: 'max', ceiling_source: 'model', declared: 262144, predicted_fit: 155648,
               started_at: 155648, loaded: 155648 },
  },
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderPanel(model: ModelInfo, onLoadOnce = vi.fn()) {
  render(<ContextCeilingPanel model={model} canLoad onLoadOnce={onLoadOnce} />, { wrapper });
  return onLoadOnce;
}

beforeEach(() => {
  api.setContextCeiling.mockReset().mockResolvedValue(GGUF);
  api.load.mockReset().mockResolvedValue(LOADED);
});

describe('ContextCeilingPanel', () => {
  it('shows the server default when the model has no ceiling', () => {
    renderPanel(GGUF);
    expect(screen.getByTestId('context-ceiling-now')).toHaveTextContent(
      "Ceiling: the server's default (GGUF_CONTEXT_LENGTH).");
  });

  it('shows what the loaded model got and why', () => {
    renderPanel(LOADED);
    expect(screen.getByTestId('context-ceiling-now')).toHaveTextContent(
      "Ceiling: as large as fits on the card, up to what the model declares (this model's setting).");
    expect(screen.getByTestId('context-loaded')).toHaveTextContent(
      "Loaded with a 155,648-token window and a q8_0 key-value (KV) cache; the ceiling was this model's setting.");
    expect(screen.getByTestId('context-kv-reason')).toHaveTextContent('KV cache: q8_0: GGUF_KV_CACHE_TYPE');
  });

  it('sends max', async () => {
    renderPanel(GGUF);
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'max');
    await userEvent.click(screen.getByTestId('context-ceiling-save'));
    await waitFor(() => expect(api.setContextCeiling).toHaveBeenCalledTimes(1));
    expect(api.setContextCeiling).toHaveBeenCalledWith(7, 'max');
  });

  it('sends a number of tokens', async () => {
    renderPanel(GGUF);
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'tokens');
    await userEvent.type(screen.getByLabelText('Context ceiling in tokens'), '131072');
    await userEvent.click(screen.getByTestId('context-ceiling-save'));
    await waitFor(() => expect(api.setContextCeiling).toHaveBeenCalledWith(7, 131072));
  });

  it('sends null to go back to the server default', async () => {
    renderPanel(LOADED);
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'server');
    await userEvent.click(screen.getByTestId('context-ceiling-save'));
    await waitFor(() => expect(api.setContextCeiling).toHaveBeenCalledWith(7, null));
  });

  it('refuses a window below 2,048 before sending anything', async () => {
    renderPanel(GGUF);
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'tokens');
    await userEvent.type(screen.getByLabelText('Context ceiling in tokens'), '1024');
    expect(screen.getByTestId('context-ceiling-save')).toBeDisabled();
    expect(screen.getByText('Give at least 2,048 tokens.')).toBeInTheDocument();
    expect(api.setContextCeiling).not.toHaveBeenCalled();
  });

  it('loads once with the chosen ceiling', async () => {
    const onLoadOnce = renderPanel(GGUF);
    expect(screen.queryByTestId('context-ceiling-load-once')).toBeNull();
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'max');
    await userEvent.click(screen.getByTestId('context-ceiling-load-once'));
    expect(onLoadOnce).toHaveBeenCalledWith(7, 'max');
    expect(api.setContextCeiling).not.toHaveBeenCalled();
  });
});

describe('the wiring', () => {
  it('the details modal shows it for a GGUF model only, and its load-once reaches the page callback', async () => {
    const onLoadWithCeiling = vi.fn();
    const { unmount } = render(
      <ModelDetailsModal model={GGUF} isOpen onClose={vi.fn()} onLoadWithCeiling={onLoadWithCeiling} />,
      { wrapper },
    );
    await userEvent.selectOptions(screen.getByLabelText('Context ceiling'), 'max');
    await userEvent.click(screen.getByTestId('context-ceiling-load-once'));
    expect(onLoadWithCeiling).toHaveBeenCalledWith(7, 'max');
    unmount();
    render(<ModelDetailsModal model={{ ...GGUF, gguf_label: '' }} isOpen onClose={vi.fn()} />, { wrapper });
    expect(screen.queryByTestId('context-ceiling')).toBeNull();
  });

  it('the hook passes the ceiling to the API', async () => {
    const { result } = renderHook(() => useModels(), { wrapper });
    result.current.load({ id: 7, contextCeiling: 'max' });
    await waitFor(() => expect(api.load).toHaveBeenCalledTimes(1));
    expect(api.load.mock.calls[0][0]).toBe(7);
    expect(api.load.mock.calls[0][2]).toBe('max');
    result.current.load(7);
    await waitFor(() => expect(api.load).toHaveBeenCalledTimes(2));
    expect(api.load.mock.calls[1][2]).toBeUndefined();
  });
});

afterEach(() => vi.unstubAllGlobals());
