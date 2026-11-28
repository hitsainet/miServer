/**
 * `modelApi.load` puts the chosen card in the request body.
 *
 * MUTATION CONTROL: drop `body` from modelApi.load -> both tests fail, and every
 * load would silently go to Auto.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import { modelApi } from '../api';

function stubFetch() {
  const fetchMock = vi.fn(async () =>
    new Response(JSON.stringify({ success: true, data: { id: 3, name: 'm' }, error: null }), {
      status: 202,
      headers: { 'content-type': 'application/json' },
    }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('modelApi.load', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('sends the named card', async () => {
    const fetchMock = stubFetch();
    await modelApi.load(3, 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/models/3/load');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      gpu: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    });
  });

  it('sends auto when no card is named', async () => {
    const fetchMock = stubFetch();
    await modelApi.load(3);
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({ gpu: 'auto' });
  });
});

/**
 * The context ceiling on the wire (2026-11-21).
 *
 * MUTATION CONTROLS: drop the body from setContextCeiling, or the context_ceiling spread from
 * load -> these fail.
 */
describe('the context ceiling', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('a load once sends its ceiling beside the card', async () => {
    const fetchMock = stubFetch();
    await modelApi.load(3, 'auto', 'max');
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({ gpu: 'auto', context_ceiling: 'max' });
  });

  it.each([['max', 'max'], [131072, 131072], [null, null]] as const)(
    'setContextCeiling sends %s', async (value, sent) => {
      const fetchMock = stubFetch();
      await modelApi.setContextCeiling(3, value);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toBe('/api/models/3/context-ceiling');
      expect(init.method).toBe('PUT');
      expect(JSON.parse(init.body as string)).toEqual({ context_ceiling: sent });
    });
});
