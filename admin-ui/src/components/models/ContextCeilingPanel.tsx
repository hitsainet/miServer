/**
 * A GGUF model's context ceiling (2026-11-21): how much context it is loaded with, where that
 * comes from, what the loaded model actually got, and the controls to change it.
 *
 * The ceiling comes from the first of: a load that names one ("Load once with this ceiling"),
 * this model's setting (`PUT /api/models/{id}/context-ceiling`), the server's GGUF_CONTEXT_LENGTH.
 * "As large as fits" (`"max"`) loads the longest window the card holds, up to what the model
 * declares. Saving changes the NEXT load; the loaded window stays until then.
 *
 * On the LOADED model the same load is a reload ("Reload with this ceiling", 2026-11-21): miServer
 * unloads it, waiting for running requests, and loads it again with the new ceiling. A reload that
 * would change nothing (the ceiling it was loaded with) is refused by miServer, so the button is
 * disabled for it here.
 */

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { modelApi } from '@/services/api';
import { useToast } from '@hooks/useToast';
import type { ContextCeiling, ModelInfo } from '@/types';

/** The smallest window miServer loads (`GGUF_MIN_CONTEXT`). */
export const MIN_CONTEXT = 2048;

type Mode = 'server' | 'tokens' | 'max';

function modeOf(ceiling: number | null | undefined): Mode {
  if (ceiling === null || ceiling === undefined) return 'server';
  return ceiling === 0 ? 'max' : 'tokens';
}

/** The ceiling in words, as the operator reads it. */
export function ceilingWords(ceiling: number | null | undefined): string {
  const mode = modeOf(ceiling);
  if (mode === 'server') return "the server's default (GGUF_CONTEXT_LENGTH)";
  if (mode === 'max') return 'as large as fits on the card, up to what the model declares';
  return `${(ceiling as number).toLocaleString('en-US')} tokens`;
}

const SOURCE_WORDS: Record<string, string> = {
  request: 'named by that load',
  model: "this model's setting",
  server: "the server's default",
};

export interface ContextCeilingPanelProps {
  model: ModelInfo;
  /** Whether the model can be loaded from here now (not loaded, not busy). */
  canLoad: boolean;
  /** Whether the LOADED model can be reloaded from here now (loaded, no load running). */
  canReload?: boolean;
  /** Load this model once with a ceiling of its own (a reload when it is the loaded model). */
  onLoadOnce?: (id: number, ceiling: number | 'max') => void;
}

/** Whether `chosen` is the ceiling the model was loaded with (a reload would change nothing). */
export function sameAsLoaded(chosen: ContextCeiling, loadedCeiling: number | 'max' | undefined): boolean {
  if (loadedCeiling === undefined || chosen === null) return false;
  const norm = (v: number | 'max') => (v === 'max' || v === 0 ? 'max' : v);
  return norm(chosen) === norm(loadedCeiling);
}

export function ContextCeilingPanel({ model, canLoad, canReload, onLoadOnce }: ContextCeilingPanelProps) {
  const queryClient = useQueryClient();
  const toast = useToast();
  const stored = model.context_ceiling ?? null;
  const [mode, setMode] = useState<Mode>(modeOf(stored));
  const [tokens, setTokens] = useState<string>(stored && stored > 0 ? String(stored) : '');

  const typed = Number(tokens);
  const tokensValid = mode !== 'tokens' || (Number.isInteger(typed) && typed >= MIN_CONTEXT);
  const chosen: ContextCeiling = mode === 'server' ? null : mode === 'max' ? 'max' : typed;

  const save = useMutation({
    mutationFn: (value: ContextCeiling) => modelApi.setContextCeiling(model.id, value),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['models'] });
      toast.success('Context ceiling saved. It applies from the next load.');
    },
    onError: (error: Error) => toast.error(`Could not save the context ceiling: ${error.message}`),
  });

  const loaded = model.status === 'loaded' && model.context_length;
  const decisions = model.gguf_load;
  const reloadable = Boolean(onLoadOnce && canReload && model.status === 'loaded' && mode !== 'server');
  const unchanged = reloadable && sameAsLoaded(chosen, decisions?.context?.ceiling);

  return (
    <section data-testid="context-ceiling" className="stack-y-2 border-t border-slate-700/50 pt-3 text-sm">
      <h3 className="font-medium text-slate-200">Context window</h3>
      <p className="text-xs text-slate-300" data-testid="context-ceiling-now">
        Ceiling: {ceilingWords(stored)}
        {stored === null ? '' : " (this model's setting)"}.
      </p>
      {loaded && (
        <div className="text-xs text-slate-400 stack-y-1" data-testid="context-loaded">
          <p className="text-slate-300">
            Loaded with a {model.context_length!.toLocaleString('en-US')}-token window
            {model.kv_cache_type ? ` and a ${model.kv_cache_type} key-value (KV) cache` : ''}
            {decisions?.context?.ceiling_source
              ? `; the ceiling was ${SOURCE_WORDS[decisions.context.ceiling_source] ?? decisions.context.ceiling_source}`
              : ''}
            .
          </p>
          {decisions?.kv_cache_reason && (
            <p data-testid="context-kv-reason">KV cache: {decisions.kv_cache_reason}</p>
          )}
          {decisions?.embeddings_reason && (
            <p data-testid="context-embeddings-reason">Embeddings: {decisions.embeddings_reason}</p>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <select
          aria-label="Context ceiling"
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
          className="bg-slate-900 border border-slate-700 rounded-control"
        >
          <option value="server">Server default</option>
          <option value="tokens">A number of tokens</option>
          <option value="max">As large as fits</option>
        </select>
        {mode === 'tokens' && (
          <input
            aria-label="Context ceiling in tokens"
            type="number"
            min={MIN_CONTEXT}
            step={1024}
            className="w-28 bg-slate-900 border border-slate-700 rounded-control px-1"
            value={tokens}
            placeholder="131072"
            onChange={(e) => setTokens(e.target.value)}
          />
        )}
        <button
          type="button"
          data-testid="context-ceiling-save"
          disabled={!tokensValid || save.isPending}
          onClick={() => save.mutate(chosen)}
          className="rounded-control bg-slate-700 text-slate-100 px-2 py-1 disabled:opacity-50"
        >
          Save for this model
        </button>
        {onLoadOnce && canLoad && mode !== 'server' && (
          <button
            type="button"
            data-testid="context-ceiling-load-once"
            disabled={!tokensValid}
            onClick={() => onLoadOnce(model.id, chosen as number | 'max')}
            className="rounded-control bg-primary-700 text-white px-2 py-1 disabled:opacity-50"
          >
            Load once with this ceiling
          </button>
        )}
        {reloadable && (
          <button
            type="button"
            data-testid="context-ceiling-reload"
            disabled={!tokensValid || unchanged}
            title="Unloads this model (running requests are given time to finish) and loads it again with this ceiling"
            onClick={() => onLoadOnce!(model.id, chosen as number | 'max')}
            className="rounded-control bg-primary-700 text-white px-2 py-1 disabled:opacity-50"
          >
            Reload with this ceiling
          </button>
        )}
      </div>
      {reloadable && (
        <p className="text-xs text-slate-400" data-testid="context-ceiling-reload-note">
          {unchanged
            ? 'It is loaded with this ceiling already: a reload would change nothing.'
            : 'A reload unloads the model first: armed probes are disarmed, and requests wait until it is back.'}
        </p>
      )}
      {!tokensValid && (
        <p className="text-xs text-amber-300">Give at least {MIN_CONTEXT.toLocaleString('en-US')} tokens.</p>
      )}
      <p className="text-xs text-slate-500">
        A longer window holds longer prompts but takes more of the card, which other work shares.
        &quot;As large as fits&quot; tries several windows while loading, so the load takes longer.
      </p>
    </section>
  );
}

export default ContextCeilingPanel;
