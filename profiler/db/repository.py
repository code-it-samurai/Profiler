import aiosqlite
import json
from datetime import datetime
from profiler.db.database import DB_PATH
from profiler.models.session import SearchSession
from profiler.models.profile import Profile


async def create_session(session: SearchSession) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO sessions (id, target_name, target_type, context, status,
               narrowing_round, candidates_count, known_facts, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(session.id),
                session.target_name,
                session.target_type.value,
                session.context,
                session.status.value,
                session.narrowing_round,
                session.candidates_count,
                json.dumps(session.known_facts),
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )
        await db.commit()


async def update_session(session_id: str, **kwargs) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        sets = []
        vals = []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(json.dumps(v) if isinstance(v, dict) else v)
        sets.append("updated_at = ?")
        vals.append(datetime.utcnow().isoformat())
        vals.append(session_id)
        await db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", vals)
        await db.commit()


async def get_session(session_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if row:
            return dict(row)
    return None


async def save_profile(session_id: str, profile: Profile) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO profiles (session_id, profile_json, created_at) VALUES (?, ?, ?)",
            (session_id, profile.model_dump_json(), datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_profile(session_id: str) -> Profile | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT profile_json FROM profiles WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row:
            return Profile.model_validate_json(row[0])
    return None


async def delete_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM profiles WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
