---
title: Emergent Negotiation Arena
emoji: 🧠
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: ui/app.py
pinned: true
license: mit
---

# Emergent Negotiation Arena

Three AI agents. No shared language. They must invent one to survive.

This file is the **Hugging Face Space configuration** — the YAML frontmatter
above is required by Spaces and is deliberately kept out of the repository's
main README. When deploying to a Space, this file must be uploaded as the
Space's root `README.md` (the deploy script does this automatically).

Space runtime notes:

- `app_file: ui/app.py` serves the module-level `demo` Blocks object.
- With no `FIREWORKS_API_KEY` secret set, the Space runs the no-key demo
  backends (replay of the bundled recorded run, or live heuristic agents).
- Setting the `FIREWORKS_API_KEY` secret in Space settings enables the live
  LLM backend — note that public visitors then spend that key's credits.

Full project documentation: see the repository README.
