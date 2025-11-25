# Contributing to audio_samples

## Contributions

Contributions in the form of bug reports, feature requests, or pull requests are
welcome. For pull requests, please consider:

* Write clean and descriptive commit messages and keep the history clean.
* Ensure cargo fmt is run.
* Documentation and tests are required.

## Usage of AI Tools

Since late 2022 there has been a rapid expansion of AI-based tooling for writing and interacting with code. Large Language Models such as ChatGPT, Gemini, and Claude allow users to query codebases, debug issues, discuss architectural ideas, and even generate tests or refactor modules. Some providers now offer agentic variants that execute actions, gather results, and iteratively update their own plan in response to intermediate outputs.

The effectiveness of these tools is debated. Some developers enjoy working with them, others avoid them entirely, and most fall somewhere between. Like many people, I have experimented with a range of these systems. The Git history reflects this: at the time of writing, I have 35 commits with large-scale additions and removals, while Claude contributes 7 commits totalling around one thousand lines of change. This does not capture cases where I used Claude Code interactively without letting it commit and push autonomously, but it does show my experimentation with these tools.

---

The existence of these tools does not change the responsibilities involved in software development. **They do not excuse poor judgement, weak design, or low standards.** A contributor must still produce code that matches the project’s conventions and constraints, regardless of whether an LLM generated an initial draft. Reviewers must still apply the same scrutiny before merging. Expectations remain the same irrespective of whether AI was used during development.

More broadly, the availability of these tools removes any remaining excuse for poor project supports. If anything, the rise of LLM-assisted development makes clear documentation, tests, and examples more important, not less. They anchor the project in well-defined behaviour, provide an authoritative reference when an AI tool produces something misguided, and ensure that humans and machines alike operate against the same ground truth.

**Use them as you see fit, but ultimately you are the one responsible.**

---

In practice, I treat these systems as tools that can accelerate specific, well-defined tasks. They do not replace architectural reasoning, design decisions, or accountability. For this project, Claude Code in particular struggled with core structural elements such as ownership, trait bounds, and the architectural model that underpins ``audio_samples``. As a result, I design and implement the architecture myself, using LLMs only for contained boilerplate or refactoring where the scope is clear and the risks are manageable.

> A final note on terminology: the word AI is used here only because it has become the default label in public and technical conversations. It is not a precise description. Using it uncritically encourages anthropomorphic language and obscures the fact that these tools are fundamentally computational systems. Clear terminology matters, especially when discussing responsibility, intent, and reliability.

## Code of conduct

* Be sound.
