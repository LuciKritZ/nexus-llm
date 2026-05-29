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
    await db.commit()
