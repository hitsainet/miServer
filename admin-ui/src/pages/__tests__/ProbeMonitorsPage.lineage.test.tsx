/**
 * 043 FR-043.18: a probe weight-space programming derived from another shows its lineage on the
 * Probe Monitors page before it is armed: parent, provider, verdict and the patch (12 characters,
 * the full hash in `title`). A plain probe, and a server from before 043, show nothing.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, within } from '@testing-library/react';
import type { DerivedFrom, Probe } from '@/types/probe';

const state = { probes: [] as Probe[] };

vi.mock('@/hooks/useProbes', () => ({
  useProbes: () => ({
    probes: state.probes,
    probesLoading: false,
    status: undefined,
    statusLoading: false,
    events: [],
    eventsLoading: false,
    importProbe: vi.fn(),
    importing: false,
    arm: vi.fn(),
    arming: false,
    armingId: null,
    checkingParityId: null,
    pendingAck: undefined,
    dismissAck: vi.fn(),
    checkParity: vi.fn(),
    checkingParity: false,
    disarm: vi.fn(),
    remove: vi.fn(),
    clearEvents: vi.fn(),
  }),
  useProbeHistory: () => ({
    events: [], loaded: true, loading: false, failed: false, hasOlder: false,
    loadingOlder: false, loadOlder: vi.fn(), held: 0, reload: vi.fn(),
  }),
  MAX_LIVE_PER_PROBE: 1000,
  useProbeEventDetail: () => ({ data: undefined, isLoading: false, isError: false }),
}));

vi.mock('@/components/probes/BehaviourSets', () => ({
  BehaviourSets: () => <div data-testid="behaviour-sets-stub" />,
}));

const { ProbeMonitorsPage } = await import('../ProbeMonitorsPage');

const DERIVED: DerivedFrom = {
  parent_probe_id: 'pr_0a1b2c3d4e5f',
  parent_detector_digest: 'a'.repeat(64),
  patch_sha256: '0123456789abcdef'.repeat(4),
  provider: 'weightspace',
  provider_version: '0.3.0',
  verdict: { value: 'improved', run_sha256: 'c'.repeat(64) },
};

function probe(over: Partial<Probe> = {}): Probe {
  return {
    id: 'pr_1', name: 'high-stakes', hf_id: 'LiquidAI/LFM2.5-1.2B-Instruct', layer: 11,
    rule: 'mean', scope: 'all', basis: 'residual', streamable: true, threshold: 1.07,
    target_fpr: 0.01, rung: 0, rung_language: 'trained', next_step: 'evaluate it',
    armed: false, paused_reason: null, parity: null, created_at: '2026-11-16T00:00:00Z',
    ...over,
  } as Probe;
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ProbeMonitorsPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('the lineage line (043)', () => {
  beforeEach(() => {
    state.probes = [];
  });

  it('renders on a derived probe, naming parent, provider, verdict and the patch', () => {
    state.probes = [probe({ derived_from: DERIVED })];
    renderPage();
    const line = screen.getByTestId('probe-derived-from');
    expect(line).toHaveTextContent(
      'Derived from pr_0a1b2c3d4e5f by weightspace 0.3.0; verdict improved; patch 0123456789ab.'
    );
    expect(within(line).getByTitle(DERIVED.patch_sha256)).toHaveTextContent(/^0123456789ab$/);
  });

  it('renders nothing on a plain probe, or when the field is absent', () => {
    state.probes = [probe({ id: 'pr_a', derived_from: null }), probe({ id: 'pr_b', name: 'b' })];
    renderPage();
    expect(screen.queryByTestId('probe-derived-from')).toBeNull();
  });
});
