import aiosqlite

DB_PATH = "profiler.db"


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                target_name TEXT NOT NULL,
                target_type TEXT NOT NULL,
                context TEXT,
                status TEXT NOT NULL DEFAULT 'searching',
                narrowing_round INTEGER DEFAULT 0,
                candidates_count INTEGER DEFAULT 0,
                known_facts TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                session_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """
        )
        await db.commit()
