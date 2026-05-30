import hashlib
import json

import aiosqlite


async def sync_keys_from_json(db: aiosqlite.Connection, json_path: str) -> None:
    """Synchronizes API keys from a JSON configuration file into the SQLite database."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    platforms = data.get("platforms", {})

    # Clear existing keys to ensure no residual/deleted keys persist
    await db.execute("DELETE FROM api_keys")

    for platform, keys in platforms.items():
        for key_obj in keys:
            key_value = key_obj.get("key_value")
            priority = key_obj.get("priority", 1)
            meta = json.dumps(key_obj.get("meta", {}))

            if not key_value:
                continue

            # Create a deterministic hash of the key
            key_hash = hashlib.sha256(key_value.encode("utf-8")).hexdigest()

            # Upsert into database
            await db.execute(
                """
                INSERT INTO api_keys (platform, key_hash, key_value, priority, meta)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key_hash) DO UPDATE SET
                    priority = excluded.priority,
                    meta = excluded.meta
            """,
                (platform, key_hash, key_value, priority, meta),
            )

    await db.commit()
