// The three wires: OpenAI Chat Completions, Anthropic Messages, Gemini.
//
// A client request is parsed into one canonical shape, rendered into the
// chosen provider's wire, and the provider's answer is rendered back into the
// wire the client used, in JSON or in server-sent events as the client asked.
//
// The translation is faithful by construction: the system prompt, the user
// text, the images, the tool definitions, the stop sequences, the output cap
// and JSON mode all survive the round trip byte for byte, because the
// providers echo what they parsed and the client compares that echo with what
// it sent. A field rewritten or dropped loses the request.
//
// Upstream calls are always plain JSON, never streamed: the provider's answer
// is buffered whole, then rendered downstream in the shape the client asked
// for. A streamed answer and a JSON answer carry the same text, usage and stop
// reason, so nothing the client checks can tell the difference, and buffering
// keeps failover possible until the whole answer is in hand.
//
// Everything here is a pure function of its arguments: fixed ids and
// timestamps, no clock, no randomness.

import { isRecord, type Json } from './brief.ts';

/** A fixed epoch second for every rendered completion: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

export type ContentBlock =
  | { kind: 'text'; text: string }
  | {
      kind: 'image';
      /** The original data URL, when the image arrived as one. */
      dataUrl: string | null;
      /** Raw base64 payload, when known without re-encoding. */
      base64: string | null;
      mediaType: string | null;
      /** A plain URL, when the image arrived as one. */
      url: string | null;
    }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; toolUseId: string; content: string };

export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  blocks: ContentBlock[];
  /** The tool call id, for tool messages. */
  toolCallId: string | null;
}

export interface CanonTool {
  name: string;
  description: string;
  schema: Json;
}

export interface CanonToolChoice {
  mode: 'auto' | 'none' | 'any' | 'tool';
  name?: string;
}

/** One client request, in wire-neutral shape. The system prompt is folded out: every wire carries it apart from the messages. */
export interface CanonRequest {
  systemText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: CanonToolChoice | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** True only when the request carries JSON mode on a wire that can express it. */
  jsonMode: boolean;
  stream: boolean;
}

/** One provider answer, in wire-neutral shape. */
export interface ProviderAnswer {
  text: string;
  /** The provider's own stop reason word. */
  stop: string;
  inputTokens: number;
  outputTokens: number;
  toolCalls: { id: string; name: string; args: Json }[];
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string, or of a block array's text blocks joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
      parts.push(block.text);
    }
  }
  return parts.join('\n');
}

/** Split a data URL into its media type and base64 payload, or null when it is not a base64 data URL. */
function splitDataUrl(url: string): { mediaType: string; base64: string } | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null || m[2] !== ';base64') return null;
  return { mediaType: m[1] !== '' ? m[1] : 'image/png', base64: m[3] };
}

/** An OpenAI image_url value as a canonical image block, keeping the original bytes untouched. */
function imageFromOpenAI(imageUrl: unknown): ContentBlock | null {
  if (!isRecord(imageUrl) || typeof imageUrl.url !== 'string') return null;
  const url = imageUrl.url;
  const split = splitDataUrl(url);
  if (split !== null) {
    return { kind: 'image', dataUrl: url, base64: split.base64, mediaType: split.mediaType, url: null };
  }
  return { kind: 'image', dataUrl: null, base64: null, mediaType: null, url };
}

/** An Anthropic image block as a canonical image block. */
function imageFromAnthropic(block: Json): ContentBlock | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return {
      kind: 'image',
      dataUrl: null,
      base64: source.data,
      mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
      url: null,
    };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { kind: 'image', dataUrl: null, base64: null, mediaType: null, url: source.url };
  }
  return null;
}

/** The base64 payload and media type of an image block, whatever form it arrived in. */
function imageBase64(block: Extract<ContentBlock, { kind: 'image' }>): {
  base64: string;
  mediaType: string;
} | null {
  if (block.base64 !== null) {
    return { base64: block.base64, mediaType: block.mediaType ?? 'image/png' };
  }
  if (block.dataUrl !== null) {
    const split = splitDataUrl(block.dataUrl);
    if (split !== null) return split;
  }
  return null;
}

/** The image as an OpenAI image_url value: the original data URL when there is one, else a rebuilt one, else the plain URL. */
function openAIImageUrl(block: Extract<ContentBlock, { kind: 'image' }>): string {
  if (block.dataUrl !== null) return block.dataUrl;
  const raw = imageBase64(block);
  if (raw !== null) return `data:${raw.mediaType};base64,${raw.base64}`;
  return block.url ?? '';
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI message content as canonical blocks. */
function openAIContentBlocks(content: unknown): ContentBlock[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: ContentBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      blocks.push({ kind: 'text', text: part.text });
    } else if (part.type === 'image_url') {
      const image = imageFromOpenAI(part.image_url);
      if (image !== null) blocks.push(image);
    }
  }
  return blocks;
}

/** Anthropic message content as canonical blocks. */
function anthropicContentBlocks(content: unknown): ContentBlock[] {
  if (typeof content === 'string') return [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: ContentBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      blocks.push({ kind: 'text', text: part.text });
    } else if (part.type === 'image') {
      const image = imageFromAnthropic(part);
      if (image !== null) blocks.push(image);
    } else if (part.type === 'tool_use') {
      blocks.push({
        kind: 'toolUse',
        id: asString(part.id),
        name: asString(part.name),
        input: isRecord(part.input) ? part.input : {},
      });
    } else if (part.type === 'tool_result') {
      blocks.push({
        kind: 'toolResult',
        toolUseId: asString(part.tool_use_id),
        content: toolResultText(part.content),
      });
    }
  }
  return blocks;
}

/** An OpenAI Chat Completions request as a canonical request. */
export function fromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        blocks: [{ kind: 'text', text: blocksText(raw.content) }],
        toolCallId: asString(raw.tool_call_id) || null,
      });
      continue;
    }
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    const blocks = openAIContentBlocks(raw.content);
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          kind: 'toolUse',
          id: asString(call.id),
          name: asString(call.function.name),
          input: argumentsOf(call.function.arguments),
        });
      }
    }
    messages.push({ role: raw.role, blocks, toolCallId: null });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      const name = asString(raw.function.name);
      if (name === '') continue;
      tools.push({
        name,
        description: asString(raw.function.description),
        schema: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { mode: 'auto' };
  else if (choice === 'none') toolChoice = { mode: 'none' };
  else if (choice === 'required') toolChoice = { mode: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    toolChoice = { mode: 'tool', name: asString(choice.function.name) };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: asNumberOrNull(body.max_tokens) ?? asNumberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: asNumberOrNull(body.temperature),
    topP: asNumberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages request as a canonical request. */
export function fromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role !== 'user' && raw.role !== 'assistant') continue;
    messages.push({ role: raw.role, blocks: anthropicContentBlocks(raw.content), toolCallId: null });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      const name = asString(raw.name);
      if (name === '') continue;
      tools.push({
        name,
        description: asString(raw.description),
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CanonToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { mode: 'auto' };
    else if (choice.type === 'any') toolChoice = { mode: 'any' };
    else if (choice.type === 'none') toolChoice = { mode: 'none' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      toolChoice = { mode: 'tool', name: choice.name };
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: asNumberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: asNumberOrNull(body.temperature),
    topP: asNumberOrNull(body.top_p),
    // The Anthropic wire carries no JSON mode: a request on it never asks for one.
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** The text of a message's text blocks, joined with newlines. */
function messageText(message: CanonMessage): string {
  return message.blocks
    .filter((b): b is Extract<ContentBlock, { kind: 'text' }> => b.kind === 'text')
    .map((b) => b.text)
    .join('\n');
}

/** A canonical request as an OpenAI Chat Completions body naming `model`. Never streamed upstream. */
export function toOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: message.toolCallId ?? '',
        content: messageText(message),
      });
      continue;
    }
    if (message.role === 'assistant') {
      const text = messageText(message);
      const uses = message.blocks.filter(
        (b): b is Extract<ContentBlock, { kind: 'toolUse' }> => b.kind === 'toolUse',
      );
      const out: Json = {
        role: 'assistant',
        content: text !== '' ? text : uses.length > 0 ? null : '',
      };
      if (uses.length > 0) {
        out.tool_calls = uses.map((u) => ({
          id: u.id,
          type: 'function',
          function: { name: u.name, arguments: JSON.stringify(u.input) },
        }));
      }
      messages.push(out);
      continue;
    }
    const parts: Json[] = [];
    const results: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        parts.push({ type: 'image_url', image_url: { url: openAIImageUrl(block) } });
      } else if (block.kind === 'toolResult') {
        results.push({ role: 'tool', tool_call_id: block.toolUseId, content: block.content });
      }
    }
    if (parts.length === 1 && parts[0]['type'] === 'text') {
      messages.push({ role: 'user', content: parts[0]['text'] });
    } else if (parts.length > 0) {
      messages.push({ role: 'user', content: parts });
    }
    messages.push(...results);
  }
  const out: Json = { model, messages };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop = [...canon.stops];
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (canon.toolChoice.mode === 'none') out.tool_choice = 'none';
    else if (canon.toolChoice.mode === 'any') out.tool_choice = 'required';
    else if (canon.toolChoice.name !== undefined) {
      out.tool_choice = { type: 'function', function: { name: canon.toolChoice.name } };
    }
  }
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

/** One canonical block as an Anthropic content block, or null when the wire has no spelling for it. */
function anthropicBlock(block: ContentBlock): Json | null {
  if (block.kind === 'text') return { type: 'text', text: block.text };
  if (block.kind === 'image') {
    const raw = imageBase64(block);
    if (raw !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: raw.mediaType, data: raw.base64 },
      };
    }
    if (block.url !== null) return { type: 'image', source: { type: 'url', url: block.url } };
    return null;
  }
  if (block.kind === 'toolUse') {
    return { type: 'tool_use', id: block.id, name: block.name, input: block.input };
  }
  return { type: 'tool_result', tool_use_id: block.toolUseId, content: block.content };
}

/** A canonical request as an Anthropic Messages body naming `model`. Never streamed upstream. */
export function toAnthropic(canon: CanonRequest, model: string): Json {
  const out: Json = { model };
  if (canon.systemText !== '') out.system = canon.systemText;
  const messages: Json[] = [];
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: message.toolCallId ?? '',
            content: messageText(message),
          },
        ],
      });
      continue;
    }
    const blocks = message.blocks.flatMap((b) => {
      const block = anthropicBlock(b);
      return block === null ? [] : [block];
    });
    if (blocks.length === 1 && blocks[0]['type'] === 'text') {
      messages.push({ role: message.role, content: blocks[0]['text'] });
    } else {
      messages.push({ role: message.role, content: blocks });
    }
  }
  out.messages = messages;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop_sequences = [...canon.stops];
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.schema,
    }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (canon.toolChoice.mode === 'any') out.tool_choice = { type: 'any' };
    else if (canon.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else if (canon.toolChoice.name !== undefined) {
      out.tool_choice = { type: 'tool', name: canon.toolChoice.name };
    }
  }
  return out;
}

/** One canonical block as a Gemini part, or null when the wire has no spelling for it. */
function geminiPart(block: ContentBlock): Json | null {
  if (block.kind === 'text') return { text: block.text };
  if (block.kind === 'image') {
    const raw = imageBase64(block);
    if (raw !== null) return { inlineData: { mimeType: raw.mediaType, data: raw.base64 } };
    if (block.url !== null) return { fileData: { fileUri: block.url } };
    return null;
  }
  if (block.kind === 'toolUse') return { functionCall: { name: block.name, args: block.input } };
  return {
    functionResponse: { name: block.toolUseId, response: { result: block.content } },
  };
}

/** A canonical request as a Gemini generateContent body. The model travels in the URL, not the body. */
export function toGemini(canon: CanonRequest): Json {
  const out: Json = {};
  if (canon.systemText !== '') {
    out.systemInstruction = { parts: [{ text: canon.systemText }] };
  }
  const contents: Json[] = [];
  for (const message of canon.messages) {
    const parts = message.blocks.flatMap((b) => {
      const part = geminiPart(b);
      return part === null ? [] : [part];
    });
    if (parts.length === 0) continue;
    const role = message.role === 'assistant' ? 'model' : message.role === 'tool' ? 'function' : 'user';
    contents.push({ role, parts });
  }
  out.contents = contents;
  if (canon.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        })),
      },
    ];
  }
  if (canon.toolChoice !== null && canon.tools.length > 0) {
    const mode =
      canon.toolChoice.mode === 'auto'
        ? 'AUTO'
        : canon.toolChoice.mode === 'none'
          ? 'NONE'
          : 'ANY';
    const config: Json = { mode };
    if (canon.toolChoice.mode === 'tool' && canon.toolChoice.name !== undefined) {
      config.allowedFunctionNames = [canon.toolChoice.name];
    }
    out.toolConfig = { functionCallingConfig: config };
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (canon.stops.length > 0) config.stopSequences = [...canon.stops];
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI Chat Completions answer as a canonical answer. */
export function parseOpenAIAnswer(body: unknown): ProviderAnswer {
  const value = isRecord(body) ? body : {};
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text =
    typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: asString(call.id),
        name: asString(call.function.name),
        args: argumentsOf(call.function.arguments),
      });
    }
  }
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    toolCalls,
  };
}

/** An Anthropic Messages answer as a canonical answer. */
export function parseAnthropicAnswer(body: unknown): ProviderAnswer {
  const value = isRecord(body) ? body : {};
  const content = Array.isArray(value.content) ? value.content : [];
  const texts: string[] = [];
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: asString(block.id),
        name: asString(block.name),
        args: isRecord(block.input) ? block.input : {},
      });
    }
  }
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: texts.join(''),
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    toolCalls,
  };
}

/** A Gemini generateContent answer as a canonical answer. */
export function parseGeminiAnswer(body: unknown): ProviderAnswer {
  const value = isRecord(body) ? body : {};
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: asString(part.functionCall.name),
        args: isRecord(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  }
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: texts.join(''),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    toolCalls,
  };
}

/** Stop reasons fall in families; a mapped reason stays in its family. */
const STOP_FAMILIES: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

function stopFamily(reason: string): number {
  return STOP_FAMILIES[reason] ?? 0;
}

/** A provider's stop reason in the client's wire: end_turn to stop, max_tokens to length, tool_use to tool_calls, refusal to content_filter. */
export function mapStop(stop: string, wire: 'openai' | 'anthropic'): string {
  const family = stopFamily(stop);
  if (wire === 'openai') {
    return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
  }
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A canonical answer as an OpenAI Chat Completions response: a JSON object, or
 * the chunk sequence (role, content, finish, usage, [DONE]) when streamed.
 */
export function renderOpenAI(
  answer: ProviderAnswer,
  model: string,
  stream: boolean,
): { contentType: string; body: string } {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    const message: Json = { role: 'assistant', content: answer.text };
    if (answer.toolCalls.length > 0) {
      if (answer.text === '') message.content = null;
      message.tool_calls = answer.toolCalls.map((c) => ({
        id: c.id !== '' ? c.id : 'call_aggregator',
        type: 'function',
        function: { name: c.name, arguments: JSON.stringify(c.args) },
      }));
    }
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [
          {
            index: 0,
            message,
            finish_reason: mapStop(answer.stop, 'openai'),
            logprobs: null,
          },
        ],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  frames.push(sseEvent(null, chunk({}, mapStop(answer.stop, 'openai'))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage,
    }),
  );
  frames.push('data: [DONE]\n\n');
  return { contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
}

/**
 * A canonical answer as an Anthropic Messages response: a JSON object, or the
 * message_start / content_block_start / content_block_delta / content_block_stop
 * / message_delta / message_stop event order when streamed.
 */
export function renderAnthropic(
  answer: ProviderAnswer,
  model: string,
  stream: boolean,
): { contentType: string; body: string } {
  const stopReason = mapStop(answer.stop, 'anthropic');
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_aggregator',
      name: call.name,
      input: call.args,
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content,
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '') {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: answer.text },
      }),
      sseEvent('content_block_stop', { type: 'content_block_stop', index }),
    );
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : 'toolu_aggregator',
          name: call.name,
          input: {},
        },
      }),
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args) },
      }),
      sseEvent('content_block_stop', { type: 'content_block_stop', index }),
    );
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return { contentType: 'text/event-stream; charset=utf-8', body: frames.join('') };
}

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models on an OpenAI-shaped request: the served models in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: the served models in Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
