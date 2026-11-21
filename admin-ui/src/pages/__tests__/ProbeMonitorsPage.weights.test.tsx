/**
 * 043 FR-043.5: the Probe Monitors page says, once, when the served weights are not the base the
 * probes were fitted on (a commit, a save or a checkpoint load). Three cases: the weights differ
 * (the line renders, with origin and commit count), the base (nothing), and an older server that
 * does not send `served_weights` (nothing).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import type { ProbeStatus, ServedWeights } from '@/types/probe';

const state = { status: undefined as ProbeStatus | undefined };

vi.mock('@/hooks/useProbes', () => ({
  useProbes: () => ({
    probes: [],
    probesLoading: false,
    status: state.status,
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

function status(served_weights?: ServedWeights): ProbeStatus {
  return {
    armed: [],
    armed_count: 0,
    max_armed: 8,
    imported_count: 0,
    paused_reasons: [],
    last_request_overhead_ms: null,
    overhead_warn_threshold_ms: 5,
    events_recorded: 0,
    socket_events_dropped: 0,
    force_serial: false,
    ...(served_weights ? { served_weights } : {}),
  } as ProbeStatus;
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

describe('the served-weights line (043)', () => {
  beforeEach(() => {
    state.status = undefined;
  });

  it('renders when the weights differ, naming the origin and the commits since load', () => {
    state.status = status({
      name: 'tiny', origin: 'model', commits_since_load: 1, differs_from_base: true,
    });
    renderPage();
    const line = screen.getByTestId('served-weights-provisional');
    expect(line).toHaveTextContent(
      'Verdicts are provisional: the served weights are not the base these probes were fitted on'
    );
    expect(line).toHaveTextContent('origin model, 1 commit since load');
    expect(line).toHaveTextContent('Serve the base model again to clear this.');
  });

  it('says the count is unknown when the server could not read it', () => {
    state.status = status({
      name: 'org/tiny:run:step-1', origin: 'checkpoint', commits_since_load: null,
      differs_from_base: true,
    });
    renderPage();
    expect(screen.getByTestId('served-weights-provisional')).toHaveTextContent(
      'origin checkpoint, commits since load unknown'
    );
  });

  it('renders nothing on the base', () => {
    state.status = status({
      name: 'tiny', origin: 'model', commits_since_load: 0, differs_from_base: false,
    });
    renderPage();
    expect(screen.getByTestId('probe-status')).toBeInTheDocument();
    expect(screen.queryByTestId('served-weights-provisional')).toBeNull();
  });

  it('renders nothing for a server from before 043 (the field absent)', () => {
    state.status = status();
    renderPage();
    expect(screen.getByTestId('probe-status')).toBeInTheDocument();
    expect(screen.queryByTestId('served-weights-provisional')).toBeNull();
  });
});
