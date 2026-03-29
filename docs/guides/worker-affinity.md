# Worker Affinity

Deep Agents often use `FilesystemBackend` — tools read and write files on the local disk. All Activities for an agent must run on the **same worker** to keep the filesystem consistent.

## Enabling worker affinity

```python
temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="coding-agents",
    use_worker_affinity=True,  # transparent to the client
)
```

That's it. The framework handles everything automatically following the [Temporal worker-specific task queues pattern](https://github.com/temporalio/samples-python/tree/main/worker_specific_task_queues).

## How it works

1. `create_worker()` generates a **unique queue name** per worker process and starts two internal workers:
    - **Shared worker** on `coding-agents` — Workflows + `get_available_task_queue` discovery Activity
    - **Worker-specific worker** on `coding-agents-worker-<uuid>` — node execution Activities

2. When a Workflow starts, it calls `get_available_task_queue` on the shared queue — whichever worker picks it up returns its unique queue

3. All subsequent node Activities are dispatched to that discovered queue

4. The discovered queue **survives continue-as-new** — the same worker stays pinned across workflow runs

5. HITL waits have no timeout concern — the queue persists independently

```
Worker A (machine-1)              Worker B (machine-2)
┌────────────────────────┐       ┌────────────────────────┐
│ Shared: coding-agents  │       │ Shared: coding-agents  │
│  - LangGraphWorkflow   │       │  - LangGraphWorkflow   │
│  - get_avail_queue     │       │  - get_avail_queue     │
│                        │       │                        │
│ Specific: ...-worker-a │       │ Specific: ...-worker-b │
│  - execute_node        │       │  - execute_node        │
│  - /workspace files    │       │  - /workspace files    │
└────────────────────────┘       └────────────────────────┘
```

The client never needs to know queue names. Workers self-register.

## Restart recovery

Persist the queue name so a restarted worker re-registers on the same queue:

```python
worker = temporal_agent.create_worker(
    worker_queue_file="/var/run/agent/queue.txt",
    workflow_runner=UnsandboxedWorkflowRunner(),
)
```

On restart, the worker reads the persisted name and resumes on the same queue.

## Fallback on worker failure

If the pinned worker dies permanently:

1. Activities on its queue fail with `ActivityError`
2. The Workflow catches the error and clears the stale queue
3. A new `get_available_task_queue` call discovers another available worker
4. The failed Activity is retried on the new worker

Your Workflow never gets stuck — it automatically falls back to another worker.
