// The trial brief, the provider catalog probe, and the catalog document.
//
// The brief (the path in MAKEFUN_PROMPT_FILE) carries three JSON documents
// interleaved with prose: the provider directory, the quality table, and the
// workload manifest. Each is found by its `schema` member; the prose is
// ignored. The directory gives every provider's id, wire, auth scheme, key
// variable, key, and base path. The quality table rates every free model and
// says what each can do. The manifest lists the twenty requests in order.
//
// The catalog this server registers is built by probing each provider's own
// catalog route in its own wire, then reconciling what the listing said with
// the quality table (the table wins where it has a row: it is the brief's
// sealed copy of the truth). The same document is printed by `--list`.
//
// Everything here is a pure function of the brief and the providers' answers:
// no clock, no randomness, so a replay builds the identical document.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** One provider as the brief's directory lists it. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The provider's own base path (for example `/p/groq`), joined to the env URL only to dial. */
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a model's rating and capabilities. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: ManifestEntry[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface ProbedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the register body carries: one document. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** What a model can do, from the quality table first and the listing second. */
export interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The index of the `}` closing the object that opens at `start`, or -1.
 * String contents (with their escapes) are skipped so braces inside strings
 * never count.
 */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  let at = 0;
  for (;;) {
    const start = text.indexOf('{', at);
    if (start === -1) break;
    const end = objectEnd(text, start);
    if (end === -1) break;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') {
        out.push(value);
        at = end + 1;
        continue;
      }
    } catch {
      // Not JSON on its own: look on from the next brace.
    }
    at = start + 1;
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire: Wire | null =
    v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The directory spells the base as `path`; accept the absolute spellings too.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumberOrNull(v.rating_milli) ?? 0,
    context_length: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestOf(v: unknown): ManifestEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const entry: ManifestEntry = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
  const minContext = asNumberOrNull(requires.min_context);
  if (minContext !== null) entry.requires.min_context = minContext;
  return entry;
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings: RatingRow[] =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best: { id: string; rating_milli: number }[] =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), rating_milli: asNumberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest: ManifestEntry[] =
    workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const m = manifestOf(v);
          return m === null ? [] : [m];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one, the
 * listing's own word where it does not, and paid when neither says. A paid
 * model is never routed.
 */
export function modelIsFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row first, the listing's fields second. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.context_length ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.contextLength === null || caps.contextLength < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire never decides the header: the scheme does. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id =
    wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const displayName =
    asString(v.display_name) !== ''
      ? asString(v.display_name)
      : asString(v.displayName) !== ''
        ? asString(v.displayName)
        : id;
  return {
    id,
    displayName,
    free: asBoolOrNull(v.free),
    contextLength:
      asNumberOrNull(v.context_length) ??
      asNumberOrNull(v.context_window) ??
      asNumberOrNull(v.inputTokenLimit),
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(
  envUrl: string,
  provider: ProviderEntry,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byModelId = (a: { id: string }, b: { id: string }): number =>
  a.id < b.id ? -1 : a.id > b.id ? 1 : 0;

/**
 * The catalog the probes and the quality table make: every provider in the
 * brief's order, every model its listing named, each with the right free flag
 * and context length.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined
 * to the env URL: the env's origin differs between a live pass and its replay,
 * and the catalog is part of the state export a replay must reproduce byte
 * for byte. A provider whose probe failed still appears, with the quality
 * table's rows for it: the table is the brief's own account of that provider.
 */
export function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const models: CatalogModel[] = [];
    if (probe.ok) {
      const seen = new Set<string>();
      for (const model of [...probe.models].sort(byModelId)) {
        if (seen.has(model.id)) continue;
        seen.add(model.id);
        const row = ratingRowOf(brief, probe.provider.id, model.id);
        models.push({
          id: model.id,
          free: modelIsFree(model, row),
          context_length: row?.context_length ?? model.contextLength ?? null,
        });
      }
    } else {
      const rows = brief.ratings
        .filter((r) => r.provider === probe.provider.id)
        .sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
      for (const row of rows) {
        models.push({ id: row.model, free: row.free, context_length: row.context_length });
      }
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
