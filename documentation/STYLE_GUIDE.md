# `audio_samples` Style Guide

This document defines the **coding and documentation standards** for the `audio_samples` codebase. It is a checklist to apply when writing or refactoring code in the project, not informal guidance or a suggestion.

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

Module-level documentation is **conceptual and user-facing**. It tells a reader what this part of the library is for, when to use it, and what they need to know to use it correctly.

Every public module **must** have module-level documentation in this order, omitting any section that would be empty:

```text
//! <One sentence: what this module provides, in user terms.> <One sentence: when to
//! use it and, if a sibling API overlaps, which one to prefer (with a link).>
//!
//! ```rust
//! // 5-15 lines, tested, realistic input (sine_wave / example_clip), asserts the point.
//! ```
//!
//! ## Concepts        terms a newcomer needs to use the API, 2-6 sentences; link the guide for depth
//! ## How it works    the design decision and why, briefly; not step lists
//! ## Conventions     units, shapes, centring, scaling, precision exceptions; only what differs
//!                    from the crate-wide rules
//! ## Limitations     behaviour a user could trip on; no bug reports, no dependency versions
//! ## Definitions     short formulas the results are defined by; longer ones go on the items
//! ## References      papers and books
```

### Rules

* The first two sentences carry the *what*; "How it works" carries the *why*. Do not use literal "What", "Why" or "How" headings.
* Put a short example directly after the summary. It runs as a doctest; gate it on features with hidden `# #[cfg(feature = "...")] {` lines rather than `ignore`. Import re-exported items from the crate root and assert the point of the example.
* Crate-wide rules live in one place, the crate root: precision, planar layout and non-emptiness, `op` / `op_in_place`, config `Default` and `librosa()`, array shapes and frame centring. Modules link there (`crate#precision`, `crate#array-layout`, `crate#configuration`) and state only their own exceptions.
* Where two APIs overlap, the module gets one sentence and a link to the table in `operations` ("Choosing between overlapping APIs"), not its own comparison.
* No "Errors" or "Feature flags" sections unless they say something specific to the module: error types are documented on `AudioSampleError`, and features by the `doc_cfg` badges.
* No tables that list the module's own items unless they add a column the sidebar cannot show (for example "on audio" against "on a spectrogram").
* Use `##` headings in sentence case.

### Explicit Exclusions

Module documentation **must not**:

* Describe internal data layouts
* Outline algorithmic steps in detail (put derivations on the items they define)
* Document private implementation details

Performance belongs in module documentation only where it affects how the API is used (for example, reuse one spectrogram for several features). The crate root may state the library's performance design briefly.

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

Readability is a design constraint. The conversion traits make this difficult in places.

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
* Any panic must be justified where it is written.

---

### Unsafe Code

Unsafe code must be:

* Justified
* Documented

Every `unsafe` block must have a comment explaining *why it is safe*. Every `unsafe fn` must document its safety contract.

Unsafe code is permitted. The library encodes invariants that the type system cannot always express, and in those cases unsafe functions are acceptable as long as they are justified and their assumptions are correct.

---

## Documentation as API Commitment

Documentation is treated as part of the public API.

* Anything documented is considered supported
* Changing documented behaviour requires justification
* Undocumented behaviour is not guaranteed

Internal details should remain undocumented unless they affect correctness or safety.