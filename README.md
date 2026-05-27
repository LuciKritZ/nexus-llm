# nexus-llm

`nexus-llm` is a local API proxy designed for developers running advanced local LLMs on workstations with limited unified memory (e.g., 16GB RAM). 

It intercepts incoming JSON-RPC chat completion payloads from VS Code extensions (like Continue or Cursor) and dynamically routes them to protect your local VRAM.

## Key Features

1. **Automated VRAM Unloading:** Automatically tracks active Ollama models. When a model switch occurs, it forcefully unloads the idle model using `keep_alive: 0` before loading the new one, preventing OOM crashes.
2. **Two-Step Image-to-Text Pipeline:** Intercepts multimodal turns containing images, sends them to a customizable Vision API (defaulting to Gemini 3.1 Flash-Lite in V1) for a rich markdown description, and passes purely text-based payloads to your local coding model. This allows you to retain visual agent workflows without keeping an 11B+ vision model resident in VRAM.
3. **Active Context Compressor:** Intercepts raw HTML fetched from the web and strips layout/boilerplate tags, reducing context bloat by ~65% to keep requests under the local 8,192 token limit.
4. **Image Description Cache:** Automatically caches vision API image descriptions to disk using SHA-256 hashing to preserve API tokens across sessions.

## Quick Start

`nexus-llm` is built with Python 3.12+ and `uv`.

```bash
# Setup the project
uv sync

# Run the proxy (binds to 0.0.0.0:11444 by default)
uv run python -m nexus_llm

# Clear the image cache
uv run python -m nexus_llm --clear-cache
```

**Environment Variables Required:**
- `GEMINI_API_KEY`: Required to intercept and describe images.

## Architecture

This application is built using **FastAPI** for asynchronous routing and SSE streaming, **Pydantic** for payload validation, and **httpx** for non-blocking upstream proxying to your local Ollama instance (`http://127.0.0.1:11434`).
