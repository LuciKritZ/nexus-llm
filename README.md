# nexus-llm

`nexus-llm` is a local API proxy designed for developers running advanced local LLMs on workstations with limited unified memory. 

It intercepts incoming JSON-RPC chat completion payloads from IDE extensions (like RooCode/Continue) and intelligently routes them between local Ollama instances, Anthropic's Claude API, Google's Gemini API, and OpenRouter to protect your local VRAM and wallet.

## Key Features

1. **Intelligent Capability Routing:** Uses a deterministic profiler to calculate context length and identify images. Models are dynamically selected from `models.json` based on whether they support vision and have sufficient max context limits.
2. **Context Preservation:** When routing text-only follow-ups to text-only models, older images in the chat history are automatically stripped and replaced with `[Image: <hash>]` placeholders to preserve chat flow without sending raw bytes.
3. **OpenAI Compatibility & API Security:** Native streaming protocols (like Anthropic Messages and Gemini SSE) are intercepted, parsed, and transparently translated into strict OpenAI chunk format before streaming back to the client. All endpoints are fully secured via a local static Bearer token check.
4. **Automated VRAM Unloading:** Automatically tracks active Ollama models. When a model switch occurs, it forcefully unloads the idle model using `keep_alive: 0` before loading the new one, preventing OOM crashes.
5. **Active Context Compressor & Safe Rolling Context:** Intercepts raw HTML fetched from the web and strips layout/boilerplate tags. It uses a `system_fallback` model to summarize conversational histories without dropping critical active context.
6. **Image Description Cache:** Automatically caches image payloads to disk using SHA-256 hashing.
7. **Gatekeeper Profiling:** Deterministically profiles prompt complexity. The `nexus-auto` and `auto` aliases route queries to the most appropriate, available model defined in your `platforms.json`.
8. **Failover Multiplexer & Key Rotation:** Intercepts streaming responses and seamlessly hot-swaps API keys and platforms on-the-fly (e.g., automatically recovering from 429 Resource Exhausted errors) ensuring uninterrupted E2E generation across multiple providers.

## Quick Start

`nexus-llm` is built with Python 3.12+ and `uv`.

```bash
# Setup the project
uv sync

# Run the proxy (binds to 0.0.0.0:11444 by default)
uv run nexus-llm

# Clear the image cache
uv run nexus-llm --clear-cache
```

## Background Service (macOS)

You can run `nexus-llm` silently in the background as a launchd service:

```bash
./install_service.sh
```

This will automatically create a `LaunchAgent` and add the following helpful aliases to your `~/.zshrc` or `~/.bashrc`:
- `nexus-llm-start`: Starts the background service
- `nexus-llm-stop`: Stops the background service
- `nexus-llm-log`: Tails the background service logs

*Note: Logs are automatically saved in the `.logs/` directory inside the project root (`.logs/nexus-llm.log` and `.logs/nexus-llm.err.log`).*

## Configuration

Configuration is minimal and managed via a `.env` file in the root directory:

```env
PORT=11444
PROXY_PASSWORD=your_secure_password
```

**Client Configuration:**
Because `nexus-llm` enforces standard OpenAI-compatible API authentication, you must configure your frontend clients (Continue, RooCode, Open WebUI) to pass the `PROXY_PASSWORD` as the API Key. The client will automatically send it as an `Authorization: Bearer <PROXY_PASSWORD>` header. If the password is omitted from the `.env` file, the proxy operates openly.

**Platforms & Key Management:**
Routing relies entirely on `models.json` for model capabilities, endpoints, and API authentication.
- **`models.json`**: Features a top-level `"keys"` dictionary to securely store all your API keys across various platforms (Gemini, OpenRouter, etc.) and a `"models"` dictionary defining all supported models, their context limits, and fallback logic.

## Architecture

Built using **FastAPI** for asynchronous routing and SSE streaming, **Pydantic** for payload validation and configuration, and **httpx** for non-blocking upstream proxying. It maintains a single long-lived async HTTP client with a 300-second timeout to handle slow initial VRAM loads gracefully.
