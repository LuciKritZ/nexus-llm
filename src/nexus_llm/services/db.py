import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

import aiosqlite


async def init_db(db: aiosqlite.Connection) -> None:
    """Initializes the database schema for the Smart LLM Router."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_value TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            try_after TIMESTAMP,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            meta TEXT
        );
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_platform_priority_last_used
        ON api_keys (platform, priority DESC, last_used_at ASC);
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS client_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            prefix TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            revoked_at TIMESTAMP
        );
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_client_keys_hash
        ON client_keys (key_hash, revoked_at);
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS platforms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            api_base TEXT NOT NULL,
            protocol TEXT NOT NULL DEFAULT 'openai',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            model_id TEXT NOT NULL,
            alias TEXT,
            max_input_tokens INTEGER DEFAULT 200000,
            supports_vision BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform, model_id)
        );
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_platform
        ON models (platform);
    """)

    await db.commit()


async def create_client_key(db: aiosqlite.Connection, name: str) -> tuple[str, dict[str, Any]]:
    """Generates a secure client key (nx_...), stores its hash, and returns the raw key."""
    raw_token = f"nx_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    prefix = raw_token[:10] + "..."

    await db.execute(
        """
        INSERT INTO client_keys (name, key_hash, prefix)
        VALUES (?, ?, ?)
    """,
        (name, key_hash, prefix),
    )
    await db.commit()

    return raw_token, {
        "name": name,
        "prefix": prefix,
        "key_hash": key_hash,
    }


async def list_client_keys(db: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Lists all created client keys with their statuses."""
    async with db.execute("""
        SELECT id, name, prefix, created_at, revoked_at
        FROM client_keys
        ORDER BY id DESC
    """) as cursor:
        rows = await cursor.fetchall()

    return [
        {
            "id": r[0],
            "name": r[1],
            "prefix": r[2],
            "created_at": r[3],
            "status": "revoked" if r[4] else "active",
            "revoked_at": r[4],
        }
        for r in rows
    ]


async def revoke_client_key(db: aiosqlite.Connection, identifier: str) -> bool:
    """Revokes a client key by ID or prefix."""
    if identifier.isdigit():
        cursor = await db.execute(
            """
            UPDATE client_keys SET revoked_at = CURRENT_TIMESTAMP
            WHERE id = ? AND revoked_at IS NULL
        """,
            (int(identifier),),
        )
    else:
        pattern = f"{identifier}%"
        cursor = await db.execute(
            """
            UPDATE client_keys SET revoked_at = CURRENT_TIMESTAMP
            WHERE prefix LIKE ? AND revoked_at IS NULL
        """,
            (pattern,),
        )
    await db.commit()
    return cursor.rowcount > 0


async def validate_client_key(db: aiosqlite.Connection, token: str) -> bool:
    """Checks if a client key is valid and not revoked."""
    key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    async with db.execute(
        """
        SELECT 1 FROM client_keys
        WHERE key_hash = ? AND revoked_at IS NULL
    """,
        (key_hash,),
    ) as cursor:
        row = await cursor.fetchone()
        return row is not None


async def save_platform_and_models(
    db: aiosqlite.Connection,
    platform: str,
    api_base: str,
    protocol: str,
    models_list: list[dict[str, Any]],
) -> None:
    """Saves or updates a platform and its associated models."""
    await db.execute(
        """
        INSERT INTO platforms (name, api_base, protocol)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            api_base = excluded.api_base,
            protocol = excluded.protocol
    """,
        (platform, api_base, protocol),
    )

    for m in models_list:
        model_id = m.get("id") or m.get("model_id", "")
        alias = m.get("alias")
        max_tokens = m.get("max_input_tokens", 200000)
        supports_vision = 1 if m.get("supports_vision", True) else 0

        await db.execute(
            """
            INSERT INTO models (platform, model_id, alias, max_input_tokens, supports_vision)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(platform, model_id) DO UPDATE SET
                alias = excluded.alias,
                max_input_tokens = excluded.max_input_tokens,
                supports_vision = excluded.supports_vision
        """,
            (platform, model_id, alias, max_tokens, supports_vision),
        )

    await db.commit()


async def get_all_platforms_and_models(db: aiosqlite.Connection) -> dict[str, Any]:
    """Retrieves all configured models and platforms as an in-memory dictionary."""
    async with db.execute("""
        SELECT
            m.platform, m.model_id, m.alias, m.max_input_tokens,
            m.supports_vision, p.api_base, p.protocol
        FROM models m
        LEFT JOIN platforms p ON m.platform = p.name
    """) as cursor:
        rows = await cursor.fetchall()

    platforms_dict: dict[str, Any] = {}
    for row in rows:
        platform, model_id, alias, max_tokens, supports_vision, api_base, protocol = row
        full_key = f"{platform}/{model_id}"
        model_info: dict[str, Any] = {
            "max_input_tokens": max_tokens or 200000,
            "supports_vision": bool(supports_vision),
            "mode": "chat",
        }
        if api_base:
            model_info["apiBase"] = api_base
        if protocol:
            model_info["protocol"] = protocol
        if alias:
            model_info["alias"] = alias

        platforms_dict[full_key] = model_info

    return platforms_dict


async def export_configuration(db: aiosqlite.Connection, export_path: str) -> dict[str, Any]:
    """Exports SQLite configuration (keys and models) to a JSON file."""
    keys_dict: dict[str, list[dict[str, Any]]] = {}
    async with db.execute("""
        SELECT platform, key_value, priority, meta
        FROM api_keys
        ORDER BY platform, priority DESC
    """) as cursor:
        key_rows = await cursor.fetchall()

    for platform, key_value, priority, meta in key_rows:
        if platform not in keys_dict:
            keys_dict[platform] = []
        meta_dict = json.loads(meta) if meta else {}
        keys_dict[platform].append(
            {
                "key_value": key_value,
                "priority": priority,
                "meta": meta_dict,
            }
        )

    platforms_data = await get_all_platforms_and_models(db)
    models_dict: dict[str, Any] = {}
    for k, v in platforms_data.items():
        models_dict[k] = {
            "max_input_tokens": v.get("max_input_tokens", 200000),
            "mode": v.get("mode", "chat"),
            "supports_vision": v.get("supports_vision", True),
        }
        if "apiBase" in v:
            models_dict[k]["apiBase"] = v["apiBase"]

    export_data = {
        "keys": keys_dict,
        "models": models_dict,
    }

    out_file = Path(export_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(export_data, indent=2), encoding="utf-8")

    return export_data


async def auto_import_from_json_if_empty(db: aiosqlite.Connection, json_path: str) -> None:
    """Imports platforms, models, and keys from models.json if the database has no models."""
    async with db.execute("SELECT COUNT(*) FROM models") as cursor:
        row = await cursor.fetchone()
        if row and row[0] > 0:
            return

    path = Path(json_path)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return

    models_data = data.get("models", {})
    platform_models: dict[str, list[dict[str, Any]]] = {}
    platform_bases: dict[str, str] = {}

    for full_name, info in models_data.items():
        if not isinstance(info, dict) or "/" not in full_name:
            continue
        platform, model_id = full_name.split("/", 1)
        if platform not in platform_models:
            platform_models[platform] = []
        platform_models[platform].append(
            {
                "model_id": model_id,
                "max_input_tokens": info.get("max_input_tokens", 200000),
                "supports_vision": info.get("supports_vision", True),
            }
        )
        if "apiBase" in info:
            platform_bases[platform] = info["apiBase"]

    # Also inspect keys for api_base and protocol
    keys_data = data.get("keys", {})
    platform_protocols: dict[str, str] = {}
    for platform, keys_list in keys_data.items():
        if isinstance(keys_list, list):
            for k in keys_list:
                if isinstance(k, dict):
                    meta = k.get("meta", {})
                    if "api_base" in meta and platform not in platform_bases:
                        platform_bases[platform] = meta["api_base"]
                    if "protocol" in meta:
                        platform_protocols[platform] = meta["protocol"]

    for platform, m_list in platform_models.items():
        api_base = platform_bases.get(platform, "")
        protocol = platform_protocols.get(platform, "openai")
        await save_platform_and_models(db, platform, api_base, protocol, m_list)
