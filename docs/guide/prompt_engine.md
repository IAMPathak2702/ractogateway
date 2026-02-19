# Prompt Engine

`RactoPrompt` is the core of RactoGateway's anti-hallucination prompting system.
It enforces the **RACTO** principle: **R**ole, **A**im, **C**onstraints, **T**one, **O**utput.

Key classes:

- `RactoPrompt` — the structured prompt model; call `.compile()` to get the system prompt string
- `RactoFile` — file attachment (images, PDFs, text) for multimodal requests

For full API details see [api/prompts](../api/prompts.md).
