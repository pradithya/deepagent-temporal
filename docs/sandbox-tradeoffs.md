# Sandbox Tradeoffs

This document explains why `deepagent-temporal` requires `UnsandboxedWorkflowRunner` and the implications for workflow replay determinism.

## Why Unsandboxed?

Temporal's Python SDK includes a workflow sandbox that prevents non-deterministic operations (I/O, random, system calls) inside workflow code. The sandbox works by restricting imports: any module that could introduce non-determinism is blocked.

`deepagent-temporal` requires `UnsandboxedWorkflowRunner()` because the LangGraph runtime imports modules that trigger sandbox violations:

- **`langchain-core`** — imports `json`, `hashlib`, `uuid`, and other modules that the sandbox flags as non-deterministic.
- **`pydantic`** — used extensively by LangChain for data validation; triggers sandbox restrictions on `datetime` and `typing` internals.
- **`langgraph` checkpoint machinery** — accesses `threading`, `asyncio`, and `collections` internals.

These imports happen at module load time, not at workflow execution time. The sandbox cannot distinguish between "this module is imported but only used in Activities" and "this module does non-deterministic things in workflow code."

## Risk Assessment

### What the Sandbox Protects Against

Temporal workflows must be **deterministic** — replaying the same workflow from its Event History must produce the same sequence of Activity/timer/signal commands. Non-determinism during replay causes `NonDeterminismError` and the workflow gets stuck.

Common sources of non-determinism:
- `random.random()` in workflow code
- `datetime.now()` in workflow code
- `uuid.uuid4()` in workflow code
- Network I/O in workflow code

### Why the Risk is Low in This Architecture

In `deepagent-temporal`, the workflow function (`LangGraphWorkflow.run`) is a **thin dispatcher**:

1. It reads the graph topology (node names, edges, conditional edges).
2. It dispatches Activities for each node (`execute_node`).
3. It applies state updates from Activity results.
4. It handles signals, interrupts, and continue-as-new.

**All LangGraph execution happens inside Activities**, not in the workflow function. This includes:

- LLM invocations (`call_model` Activity)
- Tool execution (`tools` Activity)
- Middleware processing (summarization, prompt caching, tool call patching)
- All non-deterministic operations (UUID generation, timestamp access, I/O)

The workflow function itself only performs deterministic operations:
- Reading from `WorkflowInput` (deterministic)
- Conditional edge evaluation (dispatched as Activities)
- State channel updates (deterministic dict merges)
- Timer/signal waits (Temporal primitives, replay-safe)

### Residual Risks

Despite the safe architecture, disabling the sandbox means Temporal cannot automatically catch mistakes:

1. **Future code changes** — if someone adds a `uuid.uuid4()` call to the workflow function (not an Activity), it would cause replay non-determinism. The sandbox would catch this; without it, the bug only surfaces on replay.
2. **Import side effects** — if a LangGraph upgrade introduces module-level side effects that run during workflow replay, they could cause non-determinism.
3. **Conditional edge evaluation** — if a conditional edge function (evaluated in the workflow) uses non-deterministic values, replays may diverge. Currently, conditional edges are evaluated as Activities in `langgraph-temporal`, avoiding this risk.

## Mitigations

### 1. Architecture-Level Safety

The `langgraph-temporal` architecture already provides the strongest mitigation: all non-deterministic code runs in Activities, not in the workflow. This is the correct pattern regardless of sandbox status.

### 2. Replay Determinism Testing

Test that workflows replay correctly by running a workflow, extracting its Event History, and replaying it on a new worker. If the replay produces different Activity dispatch commands, you have a non-determinism bug.

```python
# Conceptual test (requires Temporal test utilities)
async def test_replay_determinism(temporal_client, temporal_agent):
    # Run a workflow to completion
    result = await temporal_agent.ainvoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "replay-test"}},
    )

    # Get the workflow's event history
    handle = temporal_client.get_workflow_handle("replay-test")
    history = await handle.fetch_history()

    # Replay on a fresh worker — should not raise NonDeterminismError
    replayer = WorkflowReplayer(workflows=[LangGraphWorkflow])
    await replayer.replay_workflow(history)
```

### 3. Keep Workflow Code Thin

Do not add custom logic to the workflow function. If you need custom behavior, implement it as middleware (runs in Activities) or as a custom tool (runs in Activities). Never add:

- Direct LLM calls in workflow code
- File I/O in workflow code
- Network calls in workflow code
- Random/UUID generation in workflow code

## Summary

| Aspect | Status |
|---|---|
| Sandbox mode | Disabled (`UnsandboxedWorkflowRunner`) |
| Why | LangGraph/LangChain imports trigger sandbox restrictions |
| Risk level | Low — all non-deterministic code runs in Activities |
| Mitigation | Architecture (thin workflow), replay testing |
| Future | May revisit if Temporal Python SDK adds import-allowlisting |
