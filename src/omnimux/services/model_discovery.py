import hashlib
import json
import logging
from typing import Any

import aiosqlite
import httpx

from omnimux.services.db import get_all_platforms_and_models, save_platform_and_models
from omnimux.utils.cli_checklist import run_checklist

logger = logging.getLogger(__name__)


async def fetch_upstream_models(
    api_base: str, api_key: str, protocol: str = "openai"
) -> list[dict[str, Any]]:
    """Queries an upstream endpoint to discover available models."""
    headers: dict[str, str] = {}
    if protocol == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    clean_base = api_base.rstrip("/")
    url = f"{clean_base}/v1/models" if not clean_base.endswith("/v1") else f"{clean_base}/models"

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch models from {url}: "
                f"status {response.status_code} ({response.text[:200]})"
            )
        data = response.json()

    models_list: list[dict[str, Any]] = []
    if isinstance(data, dict):
        raw_list = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    for item in raw_list:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
            if model_id:
                models_list.append(
                    {
                        "id": model_id,
                        "name": item.get("name", model_id),
                        "max_input_tokens": item.get("context_length")
                        or item.get("max_input_tokens")
                        or 200000,
                    }
                )
        elif isinstance(item, str):
            models_list.append({"id": item, "name": item, "max_input_tokens": 200000})

    return models_list


async def run_add_model_wizard(
    db: aiosqlite.Connection,
    platform: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    protocol: str | None = None,
) -> None:
    """Interactive CLI wizard for adding an upstream platform and selecting its models."""
    print("\n--- Omnimux Model Setup Wizard ---")
    if not platform:
        platform = input("Platform name (e.g. openrouter, tokenharbor, custom): ").strip()
    if not protocol:
        prot_input = input("Protocol [1: openai-compatible (default), 2: anthropic]: ").strip()
        protocol = "anthropic" if prot_input == "2" else "openai"
    if not api_base:
        default_base = (
            "https://openrouter.ai/api"
            if platform == "openrouter"
            else "https://api.anthropic.com"
            if protocol == "anthropic"
            else "http://127.0.0.1:11434"
        )
        base_input = input(f"API Base URL [{default_base}]: ").strip()
        api_base = base_input or default_base
    if not api_key:
        api_key = input("API Key: ").strip()

    print(f"\nDiscovering models from {api_base}...")
    try:
        discovered = await fetch_upstream_models(api_base, api_key, protocol)
    except Exception as e:
        print(f"Error fetching models: {e}")
        manual_id = input(
            "Enter model ID manually to add anyway (or press Enter to abort): "
        ).strip()
        if not manual_id:
            return
        discovered = [{"id": manual_id, "name": manual_id, "max_input_tokens": 200000}]

    if not discovered:
        print("No models found from endpoint.")
        return

    options = [(m["id"], m.get("name", m["id"])) for m in discovered]
    selected_ids = set(run_checklist(options, prompt="Select models to register in Omnimux:"))

    if not selected_ids:
        print("No models selected. Aborted.")
        return

    selected_models = [m for m in discovered if m["id"] in selected_ids]

    # Save platform and models
    await save_platform_and_models(db, platform, api_base, protocol, selected_models)

    # Save API key
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    meta_json = json.dumps({"api_base": api_base, "protocol": protocol})
    await db.execute(
        """
        INSERT INTO api_keys (platform, key_hash, key_value, priority, meta)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(key_hash) DO UPDATE SET
            priority = excluded.priority,
            meta = excluded.meta
    """,
        (platform, key_hash, api_key, meta_json),
    )
    await db.commit()

    print(f"\nSuccessfully added platform '{platform}' with {len(selected_models)} model(s)!\n")


async def list_configured_models(db: aiosqlite.Connection) -> None:
    """Prints a formatted summary of all models registered in the database."""
    platforms_data = await get_all_platforms_and_models(db)
    if not platforms_data:
        print("\nNo models configured yet. Run 'omnimux model add' to add a platform.\n")
        return

    print("\n--- Configured Omnimux Models ---")
    print(f"{'Model Name':<50} {'Context':<10} {'Vision':<8} {'Base URL'}")
    print("-" * 90)
    for model_name, info in platforms_data.items():
        ctx = str(info.get("max_input_tokens", 200000))
        vis = "Yes" if info.get("supports_vision") else "No"
        base = info.get("apiBase", "default")
        print(f"{model_name:<50} {ctx:<10} {vis:<8} {base}")
    print()
