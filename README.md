# nexus-llm

`nexus-llm` is a local API proxy designed for developers running advanced local LLMs on workstations with limited unified memory. 

It intercepts incoming JSON-RPC chat completion payloads from IDE extensions (like RooCode/Continue) and intelligently routes them between local Ollama instances and Google's Gemini API to protect your local VRAM.

## Key Features

1. **Intelligent Multimodal Routing:** If the latest user prompt contains an image, the entire request is routed to Gemini API. If the latest prompt is text-only, it routes to your local Ollama instance.
2. **Context Preservation:** When routing text-only follow-ups to Ollama, older images in the chat history are automatically stripped and replaced with `[Image: <hash>]` placeholders. This ensures Ollama retains the chat context without choking on raw image bytes.
3. **OpenAI Compatibility:** Gemini's raw Server-Sent Events (SSE) are intercepted, parsed, and transparently translated into strict OpenAI chunk format before streaming back to the client.
4. **Automated VRAM Unloading:** Automatically tracks active Ollama models. When a model switch occurs, it forcefully unloads the idle model using `keep_alive: 0` before loading the new one, preventing OOM crashes.
5. **Active Context Compressor:** Intercepts raw HTML fetched from the web and strips layout/boilerplate tags, reducing context bloat by ~65% to keep requests under the local 8,192 token limit.
6. **Image Description Cache:** Automatically caches image payloads to disk using SHA-256 hashing.

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

Configuration is managed via a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
PORT=11444
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:9b-mlx
```

## Architecture

Built using **FastAPI** for asynchronous routing and SSE streaming, **Pydantic** for payload validation and configuration, and **httpx** for non-blocking upstream proxying. It maintains a single long-lived async HTTP client with a 300-second timeout to handle slow initial VRAM loads gracefully.
