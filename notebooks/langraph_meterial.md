# I want resources to learn LangGraph.

**Objective:** Build a structured learning path for a developer to go from zero knowledge to building production-ready LangGraph applications, covering core concepts, mechanics, patterns, and practical deployment.

---

## LangGraph Fundamentals & Core Concepts

### What LangGraph Is
LangGraph is a library for building stateful, multi-actor applications with LLMs by modeling workflows as **graphs** (directed, cyclic, or acyclic). It sits on top of LangChain but is a separate, independently versioned package (`langgraph`). While LangChain provides chains and agents as pre-built control flows, LangGraph gives you **explicit, programmable control** over the control flow: you define the graph topology, state schema, and execution logic yourself.

**Key distinction**: LangChain = opinionated abstractions (chains, agents, RAG pipelines). LangGraph = low-level graph runtime + state management + persistence. You can use LangGraph without LangChain, but they interoperate seamlessly (LangChain components become nodes).

---

### Core Abstractions

| Abstraction | Purpose | Key API |
|-------------|---------|---------|
| **StateGraph** | The graph definition. Holds the state schema (TypedDict / Pydantic), nodes, edges, and compilation settings. | `StateGraph(StateSchema)` |
| **Node** | A unit of work: a function (sync/async) that receives the current state and returns a **partial state update** (dict). Nodes can be LLMs, tools, custom logic, or subgraphs. | `graph.add_node("name", callable)` |
| **Edge** | Directs flow between nodes. Two types: **normal edges** (deterministic, always taken) and **conditional edges** (a routing function returns the next node name(s) based on state). | `graph.add_edge("a", "b")`<br>`graph.add_conditional_edges("a", router_fn)` |
| **Checkpointer** | Persists state after each node execution. Enables **pause/resume**, **human-in-the-loop**, **time-travel debugging**, and **fault tolerance**. Implementations: `MemorySaver` (in-memory), `SqliteSaver`, `PostgresSaver`, custom. | `graph.compile(checkpointer=MemorySaver())` |
| **RunnableConfig** | Passed to every node; carries `configurable` (thread_id, checkpoint_id), `callbacks`, `tags`, `metadata`. Used by checkpointer to scope persistence. | `config = {"configurable": {"thread_id": "123"}}` |

---

### Step-by-Step Execution of a Basic Graph

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. COMPILE                                                      │
│    graph = StateGraph(MyState).add_node(...).add_edge(...).    │
│    app = graph.compile(checkpointer=SqliteSaver(conn))          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. INVOKE / STREAM                                              │
│    result = app.invoke({"input": "hi"}, config={"thread_id": 1})│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CHECKPOINTER LOADS STATE                                     │
│    • If thread_id exists → loads latest checkpoint (state +     │
│      next node to run)                                          │
│    • Else → starts at graph entry point with initial state     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. NODE EXECUTION LOOP (per step)                               │
│    a) Checkpointer saves **pre-node** checkpoint (optional)    │
│    b) Node fn(state) → partial_update (dict)                   │
│

---

## State Management & Reducers Deep Dive

LangGraph state is a **TypedDict** where each field declares a reducer via `Annotated[Type, Reducer]`. The reducer decides how concurrent node writes are merged into the canonical state. If you omit a reducer, the last write wins (replace semantics).

### Annotated, TypedDict, and the reducer protocol

```python
# state_schema.py
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import add_messages  # built-in list-append reducer for message lists
from operator import add as list_concat  # generic list-append reducer
from functools import reduce

# ---------- custom dict-merge reducer ----------
def dict_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge two dicts; right overwrites left on key collision."""
    return {**left, **right}

# ---------- state schema ----------
class AgentState(TypedDict):
    # 1. Message history — uses LangGraph's add_messages (handles BaseMessage subclasses)
    messages: Annotated[List[Any], add_messages]

    # 2. Arbitrary list you want to *append* across nodes (e.g., tool call logs)
    tool_calls: Annotated[List[Dict], list_concat]

    # 3. Accumulated key/value findings — uses custom dict_merge reducer
    findings: Annotated[Dict[str, Any], dict_merge]

    # 4. Scalar field with NO reducer → last writer wins (explicit opt-out)
    current_step: str
```

**What the example teaches**  
- `add_messages` is message-aware (preserves `id`, `role`, `tool_calls`); prefer it over raw `list_concat` for chat histories.  
- `list_concat` (via `operator.add`) is the generic “append-only” reducer for any `List`.  
- `dict_merge` shows the pattern for custom reducers: a pure function `(left, right) -> merged`.  
- Omitting `Annotated` (field 4) gives replace semantics — useful for cursors, flags, or single-value checkpoints.

---

### 3-bullet cheat sheet: when to use which reducer

- **`add_messages`** — *only* for chat/message streams (`List[BaseMessage]`). Handles deduplication by `id`, preserves tool-call linkage, and integrates with LangGraph’s checkpointing.  
- **`operator.add` / `list_concat`** — any append-only list that isn’t messages: tool logs, retrieved doc IDs, audit trails. Simpler, no message semantics.  
- **Custom reducer (e.g., `dict_merge`)** — when state is a mapping that must accumulate keys from parallel branches (findings, metrics, per-user caches). Write a pure merge function; avoid side effects.

---

### Gaps in official resources (as of 202

---

## Control Flow: Edges, Conditional Branching & Subgraphs

Below is a minimal, self-contained LangGraph (v0.2+) script that demonstrates the four core control-flow primitives in ≤50 lines. Run it with `python control_flow_demo.py` after `pip install langgraph`.

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# 1️⃣  STATE DEFINITION
class State(dict):
    """Shared state passed between nodes. 'route' drives the conditional branch."""
    route: Literal["A", "B"]  # set by the router node
    payload: str              # mutated by the subgraph

# 2️⃣  SUBGRAPH — reusable component invoked as a single node
def build_subgraph() -> CompiledStateGraph:
    sg = StateGraph(State)
    sg.add_node("transform", lambda s: {"payload": s["payload"].upper()})
    sg.add_edge(START, "transform")
    sg.add_edge("transform", END)
    return sg.compile()

subgraph = build_subgraph()   # compiled once, reused like any node

# 3️⃣  MAIN GRAPH
graph = StateGraph(State)

# Start node — normal edge to router
graph.add_node("start", lambda s: {"route": "A", "payload": "hello"})

# Router node — decides next step based on state
def router(state: State) -> Literal["branch_a", "branch_b"]:
    return "branch_a" if state["route"] == "A" else "branch_b"

graph.add_node("router", router)

# Conditional edges: router → branch_a OR branch_b
graph.add_conditional_edges(
    "router",
    router,
    {"branch_a": "branch_a", "branch_b": "branch_b"},
)

# Branch A — normal node
graph.add_node("branch_a", lambda s: {"payload": s["payload"] + " from A"})

# Branch B — invokes the SUBGRAPH as a single node
graph.add_node("branch_b", subgraph)

# Both branches converge to end
graph.add_edge("branch_a", "end")
graph.add_edge("branch_b", "end")

# End node
graph.add_node("end", lambda s: s)

# Wire start → router (normal edge)
graph.add_edge(START, "start")
graph.add_edge("start", "router")

app = graph.compile()

# 4️⃣  EXECUTION
if __name__ == "__main__":
    # First run: route = "A" → branch_a
    print(app.invoke({"route": "A", "payload": ""}))
    # {'route': 'A', 'payload': 'HELLO FROM A'}

    # Second run: route = "B" → branch_b (subgraph uppercases)
    print(app.invoke({"route": "B", "payload": "hello"}))
    # {'route': 'B', 'payload': 'HELLO'}
```

### Key control-flow constructs explained

| Construct | Lines | What it does |
|-----------|-------|--------------|
| **Normal edge** | `graph.add_edge(START, "start")` | Linear, unconditional transition. |
| **Conditional edge** | `graph.add_conditional_edges("router", router, {...})` | Router function reads `state["route"]` and returns the *key* of the next node. |
| **Subgraph as node** | `graph.add_node("branch_b", subgraph)` | A compiled `StateGraph` drops in like any node; its internal edges are encapsulated. |
| **Convergence** | `graph.add_edge("branch_a", "end")` + `graph.add_edge("branch_b", "end")

---

## Common Agent Patterns in LangGraph

| Pattern | Use Case | Key Nodes/Edges | Pros | Cons | Code Sketch / Official Example |
|---------|----------|-----------------|------|------|-------------------------------|
| **ReAct** (Reasoning + Acting) | General-purpose tool-use agents; open-ended QA, coding, research | `agent` node (LLM + tools) → `tools` node (ToolNode) → conditional edge back to `agent` or `END` | Simple, well-understood; minimal graph complexity; works with any tool set | Can loop indefinitely without iteration limits; no explicit planning; struggles with multi-step reasoning requiring lookahead | `create_react_agent(model, tools)` — [official ReAct example](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) |
| **Plan-and-Execute** | Complex multi-step tasks requiring upfront planning (e.g., report generation, multi-hop research, codebase analysis) | `planner` node → `executor` node (often a ReAct subgraph) → conditional edge: `replan` / `continue` / `END` | Separates planning from execution; enables human-in-the-loop plan review; better for long-horizon tasks | Higher latency (two LLM passes per step); planner errors cascade; more graph components to debug | `create_plan_and_execute_agent(planner, executor)` — [official Plan-and-Execute example](https://langchain-ai.github.io/langgraph/how-tos/plan-and-execute/) |
| **Reflection / Self-Correction** | Tasks where output quality matters and can be verified (code generation, writing, SQL, structured extraction) | `generator` node → `reflector` node (critique) → conditional edge: `retry` (back to generator) / `accept` → `END` | Improves accuracy via iterative refinement; explicit quality gate; configurable retry budget | Adds latency and token cost per iteration; reflector can be over-critical or miss errors; needs good critique prompts | Custom graph with `generator` + `reflector` nodes — [Reflection pattern guide](https://langchain-ai.github.io/langgraph/how-tos/reflection/) |
| **Multi-Agent Handoff** (Supervisor / Swarm) | Distinct sub-tasks requiring different expertise (research → writing → review; frontend → backend → DevOps) | `supervisor` node (routes) → specialist agent nodes (each a subgraph) → edges back to `supervisor` or `END` | Modular, maintainable; agents can have different models/tools/prompts; natural separation of concerns | Supervisor becomes bottleneck; routing errors send tasks to wrong agent; state passing between agents adds complexity | `create_supervisor(agents, model)` — [Multi-Agent Supervisor example](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-supervisor/) / [Swarm example](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-swarm/) |

### Reading Order & Practical Notes

1. **Start with ReAct** — it's the building block; `create_react_agent()` is a one-liner that hides the graph construction. Read the [ReAct how-to](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) end-to-end.
2. **Move to Plan-and-Execute** — understand how the planner produces a task list and the executor (often a ReAct agent) consumes it. The [Plan-and-Execute guide](https://langchain-ai.github.io/langgraph/how-tos/plan-and-execute/) shows both the high-level API and the raw graph version; **skim the raw graph** unless you

---

## Human-in-the-Loop, Persistence & Memory

```python
# human_in_the_loop_demo.py
# Run: python human_in_the_loop_demo.py
# Requires: pip install langgraph langchain-openai python-dotenv

import os
import sqlite3
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---- State Definition ----
class AgentState(TypedDict):
    messages: Annotated[list, "Conversation history"]
    user_input: str
    approved: bool

# ---- Nodes ----
def call_model(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def human_approval(state: AgentState):
    """Interrupt point: pause execution and wait for human input."""
    # The interrupt() call serializes state and yields control
    decision = interrupt({"question": "Approve this response? (yes/no)", "draft": state["messages"][-1].content})
    return {"approved": decision.get("approved", False), "user_input": decision.get("feedback", "")}

def finalize(state: AgentState):
    if state["approved"]:
        return {"messages": [{"role": "assistant", "content": "Approved. Task complete."}]}
    else:
        return {"messages": [{"role": "assistant", "content": f"Rejected: {state['user_input']}. Please revise."}]}

# ---- Graph Construction ----
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("human_approval", human_approval)
builder.add_node("finalize", finalize)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    lambda s: "finalize",  # Always go to finalize; logic lives there
)
builder.add_edge("finalize", END)

# ---- Persistence: SqliteSaver (thread-level) ----
# Creates/uses a local SQLite file; each thread_id = one conversation session
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

# ---- Runnable Demo ----
if __name__ == "__main__":
    thread_id = "demo-thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    # First invocation: runs until interrupt
    print("=== Invocation 1: Model generates, then interrupts ===")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "Write a haiku about LangGraph"}]},
        config,
        stream_mode="values",
    ):
        print(f"State: {chunk}")

    # Simulate human approval via Command(resume=...)
    print("\n=== Resuming with approval ===")
    for chunk in graph.stream(
