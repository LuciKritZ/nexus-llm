# nexus-llm

`nexus-llm` is a database-first local API proxy and intelligent LLM router designed for developers running advanced models across local and cloud providers with unified client management.

It intercepts incoming JSON-RPC chat completion payloads and Anthropic `/v1/messages` requests from tools and IDE extensions (Claude Code, RooCode, Continue, Open WebUI) and intelligently routes them between local Ollama instances, Anthropic, Gemini, and OpenAI-compatible providers (OpenRouter, TokenHarbor, etc.).

## Key Features

1. **Database-First State with JSON Export:** Platform endpoints, upstream keys, and model registries are stored in SQLite (`nexus_llm_state.db`). Legacy `models.json` is auto-imported on first launch, and state can be exported on-demand via `nexus export <file.json>`.
2. **Interactive Model Discovery:** `nexus model add` interactively queries upstream `/v1/models` endpoints and lets you pick models to register using a terminal checklist or manual entry.
3. **Dynamic Client API Keys:** Generate application tokens with `nexus keys create [name]`. Tokens start with `nx_...` and can be inspected or revoked via `nexus keys list` and `nexus keys revoke <id>`.
4. **Detached Background Execution:** Start the proxy as a background daemon with `nexus start -d` and shut it down cleanly with `nexus stop`.
5. **Intelligent Capability Routing & Multiplexing:** Deterministically routes requests by vision support and context window, hot-swapping platforms/keys transparently upon quota limits (429) or failures.
6. **Native Anthropic & OpenAI Protocol Support:** Supports both OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` endpoints with full SSE streaming and token counting.
7. **Automated VRAM Unloading:** Automatically unloads idle local Ollama models (`keep_alive: 0`) before loading new ones to prevent out-of-memory crashes.
8. **Context Compressor & Cache:** Strips boilerplate HTML tags from web search results and caches image descriptions by SHA-256 hash.

## Installation & CLI Usage

Install directly or run via `uv`:

```bash
# Install in editable mode or into your environment
uv pip install -e .

# Or run via uvx
uvx nexus-llm --help
```

### CLI Commands

```bash
# 1. Start the proxy server (foreground)
nexus start --port 11444

# Start in detached / background mode
nexus start -d --port 11444

# Stop detached proxy server
nexus stop

# 2. Interactively discover and register upstream platforms/models
nexus model add

# List all configured models and endpoints
nexus model list

# 3. Create client API keys for tools (Claude Code, Continue, etc.)
nexus keys create claude-code
nexus keys list
nexus keys revoke <id_or_prefix>

# 4. Export database configuration to JSON
nexus export ~/Documents/nexus_config.json

# 5. Clear image cache
nexus cache --clear
```

### Client Setup (Claude Code & IDEs)

Configure Claude Code or your IDE client using a client key generated via `nexus keys create`:

```bash
# Generate a key
nexus keys create claude-code
# Example output: nx_XhfQ3XGeEezC5YBvHRjmX31CeES1b3jHXINMms1bW9Y

# Use with Claude Code
export ANTHROPIC_BASE_URL="http://127.0.0.1:11444"
export ANTHROPIC_API_KEY="nx_XhfQ3XGeEezC5YBvHRjmX31CeES1b3jHXINMms1bW9Y"
claude
```

## Configuration

Configuration can also be passed via `.env`:

```env
PORT=11444
SQLITE_DB_PATH=nexus_llm_state.db
PROXY_PASSWORD=optional_fallback_password
```

If client keys exist in SQLite, incoming requests must provide an active `nx_...` key (or `PROXY_PASSWORD` if configured) via `Authorization: Bearer <token>` or `x-api-key: <token>`.
