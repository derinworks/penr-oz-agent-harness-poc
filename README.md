# penr-oz-agent-harness-poc

**Agent = Model + Harness.**

A proof of concept for the second half of that equation. The model is bought, not built:
it arrives already able to reason, plan, and write code. What it cannot do is act, or
remember, or be checked. The harness is everything that closes that gap — and it is the
half you actually design.

## The pitch

A model call is a pure function. Text goes in, text comes out. It cannot open a file, run
a test, or recall what happened on the previous call. It has no way to know whether what
it just produced is correct, and no way to stop itself when it is looping.

Wrap that same function in something that can act on the world, decide what it gets to
see, remember across calls, check its output, and stop when it should — and it becomes an
agent. That wrapper is the harness.

The interesting consequence: **most of an agent's observed quality is harness quality.**
Swapping in a stronger model raises the ceiling; the harness decides how close you get to
it. A weak harness on a strong model produces an agent that hallucinates file paths,
forgets its own earlier conclusions, silently truncates its context, retries a failing
command forever, and reports success it never verified. None of those are model failures.
All of them are harness failures, and every one is fixable in code you own.

This repository builds that wrapper explicitly, one concern at a time, so that each
decision is something you can point at, inspect, and change.

## Who owns what

The split matters more than any individual component, so it is worth being blunt about it.

### The model owns

- **Reasoning and planning** — decomposing a task, choosing an approach, deciding which
  action comes next.
- **Language and code generation** — producing the diff, the query, the explanation.
- **In-context judgment** — reading what it was given and drawing conclusions from it.

### The harness owns

- **Acting** — the tools that exist at all, what they are allowed to touch, and the
  environment they run in.
- **Perception** — what the model sees. Every token in the context window is there
  because the harness put it there.
- **Memory** — anything that must survive a call boundary. The model contributes nothing
  here; between calls it is stateless.
- **Interpretation** — turning generated text into a validated, executable action, and
  rejecting it when it is malformed.
- **Control** — whether to continue, retry, escalate, or stop, and how much budget the
  whole thing may burn.
- **Verification** — deciding whether the work is actually done, by a standard the model
  does not get a vote on.
- **Accountability** — the trace, the cost, the artifact you hand to the next person.

The line to hold: **the model proposes, the harness disposes.** Any time the harness
trusts the model's own claim about something it could have checked — that a file exists,
that a test passed, that the task is complete — that is a bug in the harness, not a
limitation of the model.

## Scope: this is not the agentic-graph project

These are two different questions, and conflating them is the usual way agent
architectures turn to mush.

|  | [agent-graph](https://github.com/derinworks/penr-oz-agent-graph-pr-review) | this repo |
| --- | --- | --- |
| Answers | "Which step runs next?" | "What does one step get to see, do, and be judged by?" |
| Unit of work | A node in a graph | A single turn of the agent loop |
| Primitives | Nodes, edges, routers, typed state, checkpoints | Tools, environment, context assembly, memory, verification gates |
| Failure it prevents | A run that loops forever or branches wrongly | A turn that acts on the wrong context, or reports unverified success |

The agentic-graph project is about **topology** — the shape of a multi-step workflow and
the rules for moving through it. This project is about **depth** — what has to be true
inside one step for it to be worth trusting.

They compose cleanly, and deliberately so. A harness can run inside a graph node, or in a
plain `while` loop, or once. A graph can orchestrate harnesses that were built without
knowing a graph existed. Neither depends on the other, and neither is a substitute for the
other: a perfectly routed graph of untrustworthy steps is still untrustworthy, and a
flawless single step still needs something to decide what happens after it.

## What a harness is made of

Each concern below is developed as its own piece, so it can be reasoned about — and got
wrong — in isolation rather than as a tangle.

**Foundation.** The claim this repo is making, and a manifest that names the components a
given harness instance is built from, so its design is inspectable instead of implied.

**Agent–computer interface.** The tool and action interface the model is offered, the
environment those actions execute in, and the parsing that turns generated text into a
validated action — or refuses it.

**Context and state.** Prompts kept as external artifacts rather than buried in source; an
explicit policy for what stays in context versus what is retrieved on demand, and how to
truncate on overflow without destroying meaning; and the memory blocks that carry state
across calls.

**Control and orchestration.** The execution loop that decides continue-or-stop, and the
retry and failure policy that decides what a failed step earns — another attempt, a
different approach, or an escalation.

**Verification and quality.** The gates that decide whether work is actually done, and the
structured artifacts a finished run hands off.

**Ops and economics.** Tracing that makes a run reconstructable after the fact, token
accounting that makes its cost visible, and persistence that lets an interrupted run
resume instead of restart.

**Developer experience and validation.** A worked end-to-end task, a self-check that a
harness can run against its own manifest, and a benchmark to tell whether a change to any
of the above actually helped.

## Status

Early. This README is the first deliverable: the frame the rest is built into. The
components above land incrementally, each with its own design notes and tests.

## License

[MIT](LICENSE)
