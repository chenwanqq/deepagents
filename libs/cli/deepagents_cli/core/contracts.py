"""Contracts for CLI core runtime and ACP-style JSON-RPC transport."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "text_delta",
    "tool_call_started",
    "tool_call_finished",
    "approval_required",
    "token_usage",
    "error",
    "done",
]

DecisionType = Literal["approve", "reject", "edit"]


class Decision(BaseModel):
    """Human-in-the-loop decision submitted by a client."""

    type: DecisionType
    message: str | None = None
    args: dict[str, Any] | None = None


class DecisionBatch(BaseModel):
    """A list of decisions for one interrupt request."""

    decisions: list[Decision]


class SessionCreateRequest(BaseModel):
    """Session creation request payload."""

    assistant_id: str = "agent"
    model_name: str | None = None
    model_params: dict[str, Any] | None = None
    sandbox_type: str = "none"
    sandbox_id: str | None = None
    sandbox_setup: str | None = None
    auto_approve: bool = False
    enable_shell: bool = True
    thread_id: str | None = None
    cwd: str | None = None


class SessionCreateResponse(BaseModel):
    """Session creation response payload."""

    session_id: str
    thread_id: str


class SubmitMessageRequest(BaseModel):
    """Message submission request payload."""

    message: str


class SessionEvent(BaseModel):
    """Single runtime event emitted by the core."""

    event_id: int
    event_type: EventType
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ThreadInfoModel(BaseModel):
    """Serializable thread info model."""

    thread_id: str
    agent_name: str | None = None
    updated_at: str | None = None
    message_count: int | None = None


class ListThreadsResponse(BaseModel):
    """Response payload for thread listing."""

    threads: list[ThreadInfoModel]


class HealthResponse(BaseModel):
    """Health probe response."""

    status: Literal["ok"] = "ok"
    version: str


JsonRpcId = str | int | None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: JsonRpcId = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


class JsonRpcResponseIn(BaseModel):
    """Inbound JSON-RPC response from client to server."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


class JsonRpcNotification(BaseModel):
    """JSON-RPC 2.0 notification."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class SessionNewParams(BaseModel):
    """Params for `session/new`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    assistant_id: str = Field(default="agent", alias="assistantId")
    model_name: str | None = Field(default=None, alias="modelName")
    model_params: dict[str, Any] | None = Field(default=None, alias="modelParams")
    sandbox_type: str = Field(default="none", alias="sandboxType")
    sandbox_id: str | None = Field(default=None, alias="sandboxId")
    sandbox_setup: str | None = Field(default=None, alias="sandboxSetup")
    auto_approve: bool = Field(default=False, alias="autoApprove")
    enable_shell: bool = Field(default=True, alias="enableShell")
    thread_id: str | None = Field(default=None, alias="threadId")
    cwd: str | None = None


class SessionPromptParams(BaseModel):
    """Params for `session/prompt`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")
    prompt: str
    stream: bool = True


class SessionCancelParams(BaseModel):
    """Params for `session/cancel`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")


class SessionSetDescriptionParams(BaseModel):
    """Params for `session/set_description`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")
    description: str


class DecisionParams(BaseModel):
    """Params for `_deepagents/decision`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")
    interrupt_id: str = Field(alias="interruptId")
    decisions: list[Decision]


PermissionOptionKind = Literal["allow_once", "allow_always", "reject_once"]


class PermissionOption(BaseModel):
    """Permission option rendered by ACP-compatible clients."""

    id: str
    kind: PermissionOptionKind
    name: str
    description: str | None = None


class RequestPermissionRequestParams(BaseModel):
    """Params for `session/request_permission`."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    session_id: str = Field(alias="sessionId")
    tool_call: dict[str, Any] = Field(alias="toolCall")
    options: list[PermissionOption]
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class RequestPermissionOutcomeSelected(BaseModel):
    """Selected option outcome."""

    outcome: Literal["selected"] = "selected"
    option_id: str = Field(alias="optionId")


class RequestPermissionOutcomeCancelled(BaseModel):
    """Cancelled outcome."""

    outcome: Literal["cancelled"] = "cancelled"


class RequestPermissionResponseResult(BaseModel):
    """Result payload for response to `session/request_permission`."""

    outcome: str
    option_id: str | None = Field(default=None, alias="optionId")


class SessionListRequestParams(BaseModel):
    """Params for `session/list`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    cwd: str | None = None
    cursor: str | None = None


class SessionInfo(BaseModel):
    """ACP SessionInfo payload."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")
    cwd: str
    title: str | None = None
    updated_at: str | None = Field(default=None, alias="updatedAt")
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class SessionListResult(BaseModel):
    """Result payload for `session/list`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    sessions: list[SessionInfo]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class SessionDeleteRequestParams(BaseModel):
    """Params for `session/delete`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")


class SessionDeleteResult(BaseModel):
    """Result payload for `session/delete`."""

    ok: bool = True


class SessionLoadParams(BaseModel):
    """Params for `session/load`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")
    cwd: str | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list, alias="mcpServers")
    runtime_session_id: str | None = Field(default=None, alias="runtimeSessionId")
    stream: bool = True


class SessionCompactParams(BaseModel):
    """Params for `_deepagents/session_compact`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    session_id: str = Field(alias="sessionId")


class AgentsResetParams(BaseModel):
    """Params for `_deepagents/agents_reset`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    agent_name: str = Field(alias="agentName")
    source_agent: str | None = Field(default=None, alias="sourceAgent")


class SkillsListParams(BaseModel):
    """Params for `_deepagents/skills_list`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    agent: str = "agent"
    project: bool = False


class SkillsCreateParams(BaseModel):
    """Params for `_deepagents/skills_create`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    agent: str = "agent"
    project: bool = False


class SkillsInfoParams(BaseModel):
    """Params for `_deepagents/skills_info`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    agent: str = "agent"
    project: bool = False


class SkillsDeleteParams(BaseModel):
    """Params for `_deepagents/skills_delete`."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    agent: str = "agent"
    project: bool = False
    force: bool = False
