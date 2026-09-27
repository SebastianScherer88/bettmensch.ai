# Project: Custom Pipeline Orchestrator

## Project Overview

This project implements a custom Python pipeline orchestration framework.

The framework allows users to define tasks and pipelines using decorated Python functions. Pipelines are compiled into validated execution plans and executed by a runtime scheduler.

## Architecture

Read `docs/architecture.md` for the current architecture.

## Design Principles

* Keep pipeline definitions declarative.
* Separate compilation from execution.
* Keep compiler logic independent from runtime scheduling.
* Treat task dependencies as a directed acyclic graph.
* Preserve explicit, typed IO bindings.
* Resolve materializers during compilation where possible.
* Keep artifact storage separate from serialization logic.
* Prefer simple, composable abstractions over premature generalization.
* Do not introduce features or abstractions without a concrete use case.

## Implementation Rules

* Use Python type annotations and preserve type information.
* Prefer explicit interfaces and type-safe abstractions.
* Avoid unnecessary inheritance and global mutable state.
* Do not silently fall back to unsafe serialization.
* Do not introduce new dependencies without explaining why.
* Add tests for new behavior and edge cases.
* Preserve existing public APIs unless a change is explicitly requested.

## Development Workflow

Before implementing a feature:

1. Inspect the relevant source files and tests.
2. Read the relevant architecture documentation.
3. Identify affected abstractions and interfaces.
4. Propose a concise implementation plan.
5. Identify unresolved design decisions before making assumptions.

During implementation:

* Keep changes scoped to the requested feature.
* Follow existing naming and code conventions.
* Add or update tests.
* Run relevant tests and report results.

Do not silently change architectural decisions. If the requested implementation conflicts with the documented architecture, explain the conflict and propose alternatives.

## Documentation

Read `docs/architecture.md` for the current architecture.

Read `docs/design-decisions.md` for accepted architectural decisions.

Read `docs/roadmap.md` for planned work.

When an implementation reveals a meaningful architectural decision, propose a documentation update rather than silently changing the design.
