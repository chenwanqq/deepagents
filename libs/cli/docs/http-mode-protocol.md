# DeepAgents CLI HTTP Mode Protocol (Current)

This document describes the current HTTP-mode protocol used by `deepagents-cli`.

## 1. Transport Basics

- Endpoint: `POST /acp`
- Protocol envelope: JSON-RPC 2.0
- Request body types accepted by server:
  - JSON-RPC request (has `method`)
  - JSON-RPC response (no `method`, has `id` + `result/error`) for permission replies

### Content negotiation

- Non-streaming response: `application/json`
- Streaming response (NDJSON): `application/x-ndjson`
  - One JSON-RPC message per line
  - No SSE framing (`event:` / `data:`)

## 2. JSON-RPC Envelope

### Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "session/new",
  "params": {}
}
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {}
}
```

### Error

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {}
  }
}
```

## 3. Methods and Parameters

## 3.1 `initialize`

Purpose: capability discovery.

Params:

```json
{
  "clientInfo": {
    "name": "deepagents-cli"
  }
}
```

Result (key fields):

- `protocolVersion`
- `serverInfo`
- `agentCapabilities.streaming`
- `agentCapabilities.loadSession`
- `agentCapabilities.methods`
- `agentCapabilities.sessionCapabilities.list`
- `agentCapabilities.sessionCapabilities.delete`

## 3.2 `session/new`

Purpose: create a runtime session.

Params:

```json
{
  "assistantId": "agent",
  "modelName": "anthropic:claude-sonnet-4-5",
  "modelParams": {},
  "sandboxType": "none",
  "sandboxId": null,
  "sandboxSetup": null,
  "autoApprove": false,
  "enableShell": true,
  "threadId": "optional-thread-id",
  "cwd": "/absolute/path"
}
```

Result:

```json
{
  "sessionId": "runtime-session-id",
  "threadId": "thread-id"
}
```

## 3.3 `session/load`

Purpose: switch/rebind runtime to a thread and replay history.

Params:

```json
{
  "sessionId": "thread-id",
  "cwd": "/absolute/path",
  "mcpServers": [],
  "runtimeSessionId": "runtime-session-id",
  "stream": true
}
```

Streaming behavior (`stream=true`, NDJSON):

- emits `session/update` notifications (history replay)
- ends with final result for the same request `id`

Non-stream result (or final streamed result payload):

```json
{
  "sessionId": "thread-id",
  "ok": true
}
```

## 3.4 `session/prompt`

Purpose: submit one user prompt to a runtime session.

Params:

```json
{
  "sessionId": "runtime-session-id",
  "prompt": "hello",
  "stream": true
}
```

Streaming behavior (`stream=true`, NDJSON):

- emits `session/update` notifications
- may emit server request `session/request_permission`
- ends with final result (`final=true`)

Non-stream result:

```json
{
  "sessionId": "runtime-session-id",
  "final": true,
  "output": "..."
}
```

## 3.5 `session/cancel`

Purpose: cancel active run.

Params:

```json
{
  "sessionId": "runtime-session-id"
}
```

Result:

```json
{
  "cancelled": true
}
```

## 3.6 `session/set_description`

Purpose: set session description.

Params:

```json
{
  "sessionId": "runtime-session-id",
  "description": "text"
}
```

Result:

```json
{
  "ok": true
}
```

## 3.7 `session/list`

Purpose: list threads (ACP RFD style).

Params:

```json
{
  "cwd": "/absolute/path or null",
  "cursor": null
}
```

Notes:

- `cursor` is currently not implemented; non-null cursor returns invalid params.

Result:

```json
{
  "sessions": [
    {
      "sessionId": "thread-id",
      "cwd": "/service/current/cwd",
      "updatedAt": "ISO-8601",
      "_meta": {
        "agentName": "agent",
        "messageCount": 3
      }
    }
  ],
  "nextCursor": null
}
```

## 3.8 `session/delete`

Purpose: delete thread (idempotent success semantics).

Params:

```json
{
  "sessionId": "thread-id"
}
```

Result:

```json
{
  "ok": true
}
```

## 3.9 Permission flow: `session/request_permission`

This is a server-to-client JSON-RPC request sent in stream.

Server streamed request shape:

```json
{
  "jsonrpc": "2.0",
  "id": "permission-request-id",
  "method": "session/request_permission",
  "params": {
    "sessionId": "runtime-session-id",
    "toolCall": {
      "name": "tool",
      "args": {}
    },
    "options": [
      { "id": "allow-once", "kind": "allow_once", "name": "Allow once" },
      { "id": "allow-always", "kind": "allow_always", "name": "Allow always" },
      { "id": "reject-once", "kind": "reject_once", "name": "Reject once" }
    ],
    "_meta": {
      "interruptId": "runtime-interrupt-id",
      "actionRequests": []
    }
  }
}
```

Client must reply via `POST /acp` with JSON-RPC response (same `id`):

```json
{
  "jsonrpc": "2.0",
  "id": "permission-request-id",
  "result": {
    "outcome": "selected",
    "optionId": "allow-once"
  }
}
```

or

```json
{
  "jsonrpc": "2.0",
  "id": "permission-request-id",
  "result": {
    "outcome": "cancelled"
  }
}
```

## 3.10 DeepAgents extensions (`_deepagents/*`)

These are project extensions for capabilities not yet standardized in ACP.

### `_deepagents/session_compact`

Params:

```json
{
  "sessionId": "runtime-session-id"
}
```

Result:

```json
{
  "ok": true,
  "compactedMessages": 12,
  "tokensBefore": 9000,
  "tokensAfter": 2200
}
```

### `_deepagents/agents_list`

Params: `{}`

Result:

```json
{
  "agents": [
    {
      "name": "agent",
      "path": "/home/.../.deepagents/agent",
      "isDefault": true,
      "hasAgentsMd": true
    }
  ]
}
```

### `_deepagents/agents_reset`

Params:

```json
{
  "agentName": "agent",
  "sourceAgent": null
}
```

Result:

```json
{
  "ok": true,
  "agent": "agent",
  "path": "/home/.../.deepagents/agent"
}
```

### `_deepagents/skills_list`

Params:

```json
{
  "agent": "agent",
  "project": false
}
```

Result:

```json
{
  "skills": [
    {
      "name": "skill-name",
      "description": "...",
      "path": "/.../SKILL.md",
      "source": "built-in|user|project"
    }
  ]
}
```

### `_deepagents/skills_create`

Params:

```json
{
  "name": "skill-name",
  "agent": "agent",
  "project": false
}
```

Result:

```json
{
  "ok": true,
  "name": "skill-name",
  "path": "/.../skill-dir"
}
```

### `_deepagents/skills_info`

Params:

```json
{
  "name": "skill-name",
  "agent": "agent",
  "project": false
}
```

Result:

```json
{
  "skill": {
    "name": "skill-name",
    "description": "...",
    "path": "/.../SKILL.md",
    "source": "built-in|user|project"
  },
  "content": "full SKILL.md text",
  "supportingFiles": ["script.py", "README.md"]
}
```

### `_deepagents/skills_delete`

Params:

```json
{
  "name": "skill-name",
  "agent": "agent",
  "project": false,
  "force": true
}
```

Result:

```json
{
  "ok": true,
  "name": "skill-name"
}
```

## 4. Streaming `session/update` Event Payload

In NDJSON streams, `session/update` looks like:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "runtime-or-thread-id",
    "event": {
      "event_id": 1,
      "event_type": "text_delta",
      "session_id": "runtime-session-id",
      "payload": {}
    }
  }
}
```

Current `event_type` values used by the client:

- `text_delta`
- `tool_call_started`
- `tool_call_finished`
- `approval_required`
- `token_usage`
- `error`
- `done`

## 5. CLI Command to Protocol Mapping (`--transport http`)

Note:

- `--transport` and `--service-url` are global options and must appear before
  subcommands.

### Top-level interactive

- `deepagents --transport http`
  - startup: `initialize` + `session/new`
  - send message: `session/prompt` (stream)
  - thread switch/resume: `session/load` (stream)
  - cancel run: `session/cancel`
  - compact command: `_deepagents/session_compact`

### Non-interactive

- `deepagents --transport http -n "..."`
  - `initialize`
  - `session/new`
  - `session/prompt` (stream or non-stream depending on mode)

### Threads commands

- `deepagents --transport http threads list`
  - `session/list`
- `deepagents --transport http threads delete <id>`
  - `session/delete`

### Agent commands

- `deepagents --transport http list`
  - `_deepagents/agents_list`
- `deepagents --transport http reset --agent <name> [--target <src>]`
  - `_deepagents/agents_reset`

### Skills commands

- `deepagents --transport http skills list [--project]`
  - `_deepagents/skills_list`
- `deepagents --transport http skills create <name> [--project]`
  - `_deepagents/skills_create`
- `deepagents --transport http skills info <name> [--project]`
  - `_deepagents/skills_info`
- `deepagents --transport http skills delete <name> [--project] [--force]`
  - `_deepagents/skills_delete`

## 6. Scope Notes

- In HTTP mode, agents/skills/subagents are resolved on the service side filesystem context.
- `cwd` passed in `session/new` is used by the service while creating runtime session context.
- `session/list` currently has no cursor pagination implementation.
