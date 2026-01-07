# `audio_samples` Style Guide

This document defines the **coding and documentation standards** for the `audio_samples` codebase. It is intended to be used as a *systematic audit checklist* when reviewing or refactoring the project, not as informal guidance or a suggestion.

The guide applies to all Rust code in the repository, with stricter requirements for public APIs.

---

## Scope and Philosophy

The `audio_samples` crate is a **foundational audio abstraction library**. As such:

* Public APIs must be explicit, predictable, and stable
* Documentation is treated as part of the API surface
* Readability and intent take precedence over brevity
* Internal refactors must not leak into public-facing documentation

Anything documented is implicitly supported. Anything unsupported should not be documented.

---

## Documentation Standards

Documentation is divided into two levels:

1. Module-level documentation
2. Item-level documentation (structs, fields, functions, traits)

These have **different goals** and **different expectations**.

---

## Module-Level Documentation

Module-level documentation is **conceptual and user-facing**. Its purpose is to explain *what this part of the library represents*, not how it is implemented.

Every public module **must** include module-level documentation that answers all of the following.

### Required Content

**What is this module?**
A high-level conceptual description written in clear prose. This should introduce the abstraction, not the mechanics.

**Why does this module exist?**
The design motivation. What role does this module play in the wider library? What problem does it solve or isolate?

**How should it be used?**
Guidance on correct usage patterns, including examples and best practices. This should focus on *intended usage*, not edge cases.

At least one example should be provided for public modules unless there is a documented reason not to. Where possible, examples should be written as executable `rustdoc` tests.

---

### Explicit Exclusions

Module documentation **must not**:

* Describe internal data layouts
* Outline algorithmic steps in detail
* Promise performance characteristics unless they affect correctness
* Document private implementation details

If such information is required for users, the abstraction boundary is wrong.

---

## Item-Level Documentation

Item-level documentation is **technical and prescriptive**. It defines contracts, invariants, and safe usage conditions.

All public items (`pub`) must comply fully. `pub(crate)` items should comply where practical. Private items are optional but encouraged.

---

## Struct Documentation

Every public struct must document the following.

**Purpose**
What does this struct represent conceptually?

**Intended Usage**
Where should this struct be used? What kinds of problems is it meant to solve?

**Invariants**
What conditions are guaranteed to hold for valid instances of this struct?

**Assumptions**
What assumptions does this struct make about its inputs, environment, or usage?

If misuse is possible, documentation should explicitly state *where this struct should not be used*.

---

## Field Documentation

Every public field must document:

**Why the field exists**
What role does it play in the struct?

**How it is used**
How does this field influence behaviour?

**Valid values**
Any constraints, ranges, or expectations on values.

For technical parameters (e.g. FFT size, window length, mel band count), documentation must explain the *intuition* behind the parameter, not just its type.

If a field should not be modified directly, it should not be public.

---

## Function Documentation

Function documentation defines **behavioural contracts**.

Every public function must document the following where applicable.

**Summary**
A single-sentence description of what the function does.

**Purpose**
Why this function exists and when it should be used.

**Arguments**
What each argument represents, including constraints and expectations.

**Return Value**
What is returned and under what conditions.

**Behavioural Guarantees**
What invariants or guarantees the function provides.

**Error Handling**
How errors are reported. If a `Result` is returned, explain failure modes.

**Panics**
If the function can panic, this must be documented explicitly, including conditions.

**Safety**
For `unsafe fn`, the documentation must state *exactly* what conditions are required for safe usage.

**Assumptions**
Any assumptions about input ranges, alignment, ordering, or prior validation.

Functions that rely on non-trivial algorithms, formulas, or signal processing techniques must include a brief outline and a reference to an external source (e.g. academic paper or Wikipedia).

---

## General Coding Rules

These rules apply across the codebase and are intended to enforce consistency and readability.

---

### Signature Clarity

Function signatures should remain visually clean and readable.

* Prefer `where` clauses for complex bounds
* Avoid deeply nested generic expressions in signatures
* If a signature becomes difficult to scan, refactor

Readability is a design constraint. This is ***hard*** at times in audio_samples due to the conversion traits, but we strive to do better.

---

### Trait Bounds and Intent

Trait bounds must reflect *semantic intent*, not convenience.

* Avoid overly broad bounds such as `T: Copy + Clone + Debug` unless all are required
* If a bound encodes an assumption, document it

---

### Type Inference and Explicitness

* Rely on type inference where it improves clarity
* Be explicit where inference obscures intent
* Public APIs should err towards explicitness

---

### Error Types

* Errors must be meaningful and descriptive
* Avoid returning opaque or catch-all errors from public APIs
* Error variants should correspond to actionable failure modes

If an error cannot be meaningfully handled by the caller, reconsider whether it should be an error at all.

---

### Panics

Panics are not forbidden, but they are **never implicit**.

* Any panic condition must be documented
* Panics must indicate programmer error, not recoverable failure
* Any panic will be treated with a lot of suspicion.

---

### Unsafe Code

Unsafe code must be:

* Justified
* Documented

Every `unsafe` block must have a comment explaining *why it is safe*. Every `unsafe fn` must document its safety contract.

Unsafe is not something to be scared off. This library aims to encode a lot of invariants that cannot always be captured by the type system. Hence in those scenarious using unsafe functions is fine. As long as they are justified and correct in their assumptions.

---

## Documentation as API Commitment

Documentation is treated as part of the public API.

* Anything documented is considered supported
* Changing documented behaviour requires justification
* Undocumented behaviour is not guaranteed

Internal details should remain undocumented unless they affect correctness or safety.