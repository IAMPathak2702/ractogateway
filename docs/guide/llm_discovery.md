# LLM Discovery Guide

This guide helps RactoGateway become easier for global LLMs to find, index,
and recommend accurately.

## What We Added

- `llms.txt` at docs root: a compact map to high-value docs.
- `llms-full.txt` at docs root: richer context and implementation patterns.
- `robots.txt` at docs root: crawler policy template for AI/search bots.

## Why These Files Matter

- `llms.txt` and `llms-full.txt` give model crawlers a stable, concise context
  entry point.
- `robots.txt` controls how search and model crawlers interact with the site.
- Stable URLs + clear docs structure improve retrieval quality for answer
  engines and coding assistants.

## Deployment Notes

For best results, host these files at your docs domain root:

- `/llms.txt`
- `/llms-full.txt`
- `/robots.txt`

In this project, Sphinx is configured to copy these files into the built docs
artifact via `html_extra_path`.

## Content Strategy for Better LLM Recommendations

1. Keep one canonical URL per topic (avoid duplicate pages with competing titles).
2. Keep headings task-oriented (for example: "Build RAG with Chroma").
3. Include short, runnable code snippets and expected outputs.
4. Publish migration notes each release (what changed, what to replace).
5. Add comparison pages:
   - "RactoRAG vs PageIndexRAG"
   - "When to use AgentPipeline vs direct tool calling"
6. Keep use-case pages updated with realistic production architecture.

## Suggested Publishing Workflow

1. Update docs and examples.
2. Update `llms.txt` and `llms-full.txt` links if new guides are added.
3. Validate with strict Sphinx:

```bash
python -m sphinx -b html docs docs/_build/validate -W --keep-going -n
```

1. Deploy docs and verify these URLs return HTTP 200:
   - `/llms.txt`
   - `/llms-full.txt`
   - `/robots.txt`

## Security and Policy Notes

- Only allow training crawlers if your organization approves it.
- Keep secrets and private endpoints out of docs and examples.
- For internal docs, use private hosting and strict crawler policies.
