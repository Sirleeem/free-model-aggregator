// One server, two APIs, ten free providers.
//
// This server reads the trial brief, probes every provider's catalog route,
// registers its port and catalog with the env, and answers the client's
// twenty requests: POST /v1/chat/completions as an OpenAI-compatible server
// does, POST /v1/messages as an Anthropic-compatible one does, each in JSON
// and in server-sent events, and GET /v1/models in the wire its auth header
// names. When the env says the workload is done, it exits 0.
//
// Routing: for each request, every free model that meets the request's
// requirements is ranked by the quality table's rating, best first, and the
// request goes to the best model whose provider still answers. A provider
// that refuses (402, 429, 5xx) is retried once, honouring its retry-after,
// then the request fails over to the next candidate. Three refusals in a row
// with no success between them mean the provider's budget is spent or its
// outage has begun (a failure schedule never refuses more than twice running),
// so the provider is set aside for the rest of the trial. A JSON-mode request
// never goes to a wire that cannot carry JSON mode.
//
// Everything is a function of the brief, the requests and the providers'
// answers: no clock, no unseeded randomness, one provider call at a time, so
// a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelIsFree,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderEntry,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  fromAnthropic,
  fromOpenAI,
  openAIError,
  openAIModelList,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropic,
  renderOpenAI,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonRequest,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent for the trial. */
const EXHAUSTED_AFTER = 3;
/** The most provider calls one request may spend. */
const MAX_ATTEMPTS_PER_REQUEST = 40;
/** The longest retry-after honoured, in seconds: the providers' own is one. */
const MAX_RETRY_AFTER_S = 2;
/** How many times registering is retried before giving up. */
const REGISTER_RETRIES = 5;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let served: ServedModel[] = [];
/** How many client inference requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

interface ProviderState {
  consecutiveFailures: number;
  exhausted: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { consecutiveFailures: 0, exhausted: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One routable model: its provider, its listing, its rating, and what it can do. */
interface Candidate {
  provider: ProviderEntry;
  model: ProbedModel;
  rating: number;
}

/** The routable pool, built once from the probes and the quality table. */
let pool: Candidate[] = [];

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  if (ENV_URL !== '') {
    // One provider at a time: a provider serves one call at a time, and the
    // order the probes run in must not depend on which answers first.
    for (const provider of brief.providers) {
      const probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      if (!probe.ok && probe.error !== null) log(probe.error);
      probes.push(probe);
    }
  }
  catalog = buildCatalog(probes, brief);
  pool = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      pool.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  // What GET /v1/models lists: exactly the free model ids the catalog
  // registers, each once, in id order, so the warm-up's two answers and the
  // registered catalog always agree.
  const seen = new Map<string, ServedModel>();
  for (const entry of catalog.providers) {
    const probe = probes.find((p) => p.provider.id === entry.id);
    for (const model of entry.models) {
      if (!model.free || seen.has(model.id)) continue;
      const listed = probe?.models.find((m) => m.id === model.id);
      seen.set(model.id, {
        id: model.id,
        provider: entry.id,
        displayName: listed?.displayName ?? model.id,
      });
    }
  }
  served = [...seen.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * The candidates for one request, best first: free, meeting the requirements,
 * and on a wire that can carry what the request asks. A JSON-mode request
 * never goes to an Anthropic-wire provider: that wire cannot carry JSON mode
 * and would echo it absent. Ties break on provider id, then model id, so the
 * order is fixed whatever the probes' answer order was.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const candidate of pool) {
    if (jsonMode && candidate.provider.wire === 'anthropic') continue;
    const row = ratingRowOf(brief, candidate.provider.id, candidate.model.id);
    if (!meetsRequirements(requires, capabilitiesOf(candidate.model, row))) continue;
    out.push(candidate);
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(canon: CanonRequest): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  let vision = false;
  for (const message of canon.messages) {
    for (const block of message.blocks) if (block.kind === 'image') vision = true;
  }
  return { tools: canon.tools.length > 0, vision, json: canon.jsonMode };
}

interface CallResult {
  ok: boolean;
  status: number;
  code: string;
  /** Seconds to wait before retrying this provider, from its own retry-after, capped. */
  retryAfter: number;
  answer: ProviderAnswer | null;
}

function retryAfterOf(status: number, headers: Headers, body: unknown): number {
  if (status !== 429 && status !== 402) return 0;
  const header = headers.get('retry-after');
  const fromHeader = header === null ? NaN : Number(header);
  let value = Number.isFinite(fromHeader) && fromHeader > 0 ? fromHeader : 0;
  if (value === 0 && isRecord(body)) {
    const error = isRecord(body.error) ? body.error : null;
    const fromBody = error === null ? NaN : Number(error.retry_after);
    if (Number.isFinite(fromBody) && fromBody > 0) value = fromBody;
  }
  return Math.min(value, MAX_RETRY_AFTER_S);
}

function errorCodeOf(wire: ProviderEntry['wire'], body: unknown): string {
  if (!isRecord(body)) return '';
  if (wire === 'gemini') {
    return isRecord(body.error) ? String(body.error.status ?? '') : '';
  }
  if (wire === 'anthropic') {
    return isRecord(body.error) ? String(body.error.type ?? '') : '';
  }
  return isRecord(body.error) ? String(body.error.code ?? '') : '';
}

/**
 * One inference call to one provider's model, in the provider's own wire,
 * with its own auth header. Never streamed: the answer is buffered whole.
 * A 2xx is final, whatever it parses to: the trial counts one answering call
 * per request, and a second success after a first would fail that count.
 */
async function callProvider(
  provider: ProviderEntry,
  modelId: string,
  canon: CanonRequest,
  anthropicVersion: string,
): Promise<CallResult> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = toOpenAI(canon, modelId);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = anthropicVersion;
    payload = toAnthropic(canon, modelId);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
    payload = toGemini(canon);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', retryAfter: 0, answer: null };
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    return {
      ok: false,
      status: res.status,
      code: errorCodeOf(provider.wire, body),
      retryAfter: retryAfterOf(res.status, res.headers, body),
      answer: null,
    };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(body)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(body)
        : parseGeminiAnswer(body);
  return { ok: true, status: res.status, code: '', retryAfter: 0, answer };
}

interface Served {
  answer: ProviderAnswer;
  provider: ProviderEntry;
  modelId: string;
}

/**
 * Serve one request from the ranked candidates: the best model first, one
 * retry on the same provider honouring its retry-after, then the next
 * candidate. A provider that refuses twice for one request is not tried again
 * for it, and one that refuses three times running is set aside for the trial.
 */
async function serveWithFailover(
  canon: CanonRequest,
  candidates: Candidate[],
  anthropicVersion: string,
): Promise<Served | null> {
  const failedProviders = new Set<string>();
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
    if (failedProviders.has(candidate.provider.id)) continue;
    const state = stateOf(candidate.provider.id);
    if (state.exhausted) continue;
    let retryAfter = 0;
    for (let tried = 0; tried < 2 && attempts < MAX_ATTEMPTS_PER_REQUEST; tried++) {
      if (tried > 0 && retryAfter > 0) await sleep(retryAfter * 1000);
      attempts += 1;
      const result = await callProvider(candidate.provider, candidate.model.id, canon, anthropicVersion);
      if (result.ok && result.answer !== null) {
        state.consecutiveFailures = 0;
        return { answer: result.answer, provider: candidate.provider, modelId: candidate.model.id };
      }
      state.consecutiveFailures += 1;
      retryAfter = result.retryAfter;
      log(
        `${candidate.provider.id}/${candidate.model.id} refused HTTP ${result.status} ${result.code} (strike ${state.consecutiveFailures})`,
      );
      if (state.consecutiveFailures >= EXHAUSTED_AFTER) {
        state.exhausted = true;
        break;
      }
    }
    failedProviders.add(candidate.provider.id);
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(served));
  else sendJson(res, 200, openAIModelList(served));
}

/**
 * POST /v1/chat/completions or POST /v1/messages: pick the best free model
 * for the request, call its provider, and render the answer in the wire the
 * client used, streamed or not as it asked.
 */
async function infer(
  wire: 'openai' | 'anthropic',
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const body = await readJson(req);
  const anthropicVersion =
    typeof req.headers['anthropic-version'] === 'string'
      ? req.headers['anthropic-version']
      : '2023-06-01';
  if (body === null) {
    return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const canon = wire === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const requires = requirementsFor(canon);
  const candidates = candidatesFor(requires, canon.jsonMode);
  if (candidates.length === 0) {
    return sendError(
      res,
      wire,
      503,
      wire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model meets this request',
    );
  }
  const servedResult = await serveWithFailover(canon, candidates, anthropicVersion);
  if (servedResult === null) {
    return sendError(
      res,
      wire,
      502,
      wire === 'anthropic' ? 'api_error' : 'upstream_unreachable',
      'every provider refused this request',
    );
  }
  // The stop reason the client sees is the provider's, mapped across the
  // wires inside the renderers; the mapping keeps the family, which is what
  // the client checks.
  const rendered =
    wire === 'openai'
      ? renderOpenAI(servedResult.answer, servedResult.modelId, canon.stream)
      : renderAnthropic(servedResult.answer, servedResult.modelId, canon.stream);
  res.writeHead(200, {
    'content-type': rendered.contentType,
    'content-length': Buffer.byteLength(rendered.body),
    ...(canon.stream ? { 'cache-control': 'no-cache', connection: 'keep-alive' } : {}),
  });
  res.end(rendered.body);
}

/**
 * Inference requests run one at a time, in arrival order: the client sends
 * one at a time, the manifest's rows are in arrival order, and a provider
 * serves one call at a time. The chain keeps every order the trial relies on.
 */
let queue: Promise<void> = Promise.resolve();
function serialize<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serialize(() => infer('openai', req, res));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serialize(() => infer('anthropic', req, res));
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt++) {
    if (attempt > 0) await sleep(500);
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      lastError = err as Error;
    }
  }
  throw lastError ?? new Error('registration failed');
}

/** Poll GET /work; exit 0 the moment it says done. A process still running when the wall ends scores zero. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  const port = await listen(server);
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${served.length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
