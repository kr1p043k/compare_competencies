"""CRUD tests for chat_threads + chat_messages models (DB-isolated).

Isolation contract. The previous version of this file connected to the live
``DATABASE_URL`` and wiped whole tables in setup/teardown
(``DELETE FROM chat_messages; DELETE FROM chat_threads``), which destroyed live
chat history. This module must never do that again:

1. The suite runs ONLY against ``TEST_DATABASE_URL``. It never reads
   ``DATABASE_URL`` (the live/app database), so a silent fallback to it is
   impossible. Pointing the tests at the live DSN is possible only when the
   operator explicitly exports ``TEST_DATABASE_URL``.
2. ``TEST_DATABASE_URL`` unset -> the whole module skips with a clear reason.
   Host unreachable -> skips the same way.
3. The suite does not depend on an external alembic run: when the chat tables
   are missing, the session fixture creates them from the DDL of the
   ``add_chat_storage`` migration. Anything the fixture created is dropped at
   the end of the session.
4. Every test works inside a SAVEPOINT nested in an outer transaction that is
   rolled back at teardown, so nothing a test commits ever reaches the DB.
   ``test_foreign_thread_survives`` asserts this with a second connection.
5. Belt-and-suspenders cleanup tags every created row with this run's unique
   ``pytest-chat-<pid>-<uid>`` context_key and deletes only matching rows -
   never a ``DELETE`` of whole tables.
6. A committed "foreign" thread (a live-data stand-in, seeded by the fixture)
   must survive: ``test_foreign_thread_survives`` checks it mid-session and the
   session teardown fails loudly if it disappeared.
"""

from __future__ import annotations

import itertools
import os
import socket
import urllib.parse
import uuid
from collections.abc import Iterator
from typing import Any

import psycopg2
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from src.models.krm_models import ChatMessage, ChatThread

# The only DSN this module may touch. NB: deliberately NO fallback to
# DATABASE_URL - that is the live database (contract point 1).
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

# Unique per run (and per xdist worker): tags every row created here.
TAG = f"pytest-chat-{os.getpid()}-{uuid.uuid4().hex[:8]}"
_THREAD_SEQ = itertools.count(1)
_GUARD_TITLE = "Live chat history - must survive the pytest run"

# DDL copied verbatim from alembic/versions/add_chat_storage.py upgrade().
# The users stub is a minimal FK target for a fresh test database where alembic
# has not run; the fixture drops it at session end only if it created it.
_USERS_STUB_DDL = "CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY)"
_CHAT_THREADS_DDL = """
    CREATE TABLE IF NOT EXISTS chat_threads (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
        context_type    VARCHAR(20) NOT NULL
            CHECK (context_type IN ('teacher', 'student')),
        context_key     VARCHAR(255) NOT NULL,
        title           TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""
_CHAT_MESSAGES_DDL = """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        thread_id       UUID NOT NULL REFERENCES chat_threads(id)
                            ON DELETE CASCADE,
        role            VARCHAR(16) NOT NULL
            CHECK (role IN ('user', 'assistant')),
        seq             INTEGER NOT NULL,
        blocks_json     JSONB,
        model_used      VARCHAR(64),
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (thread_id, seq)
    )
"""


def _to_sync_dsn(url: str) -> str:
    """Application DSNs may carry an asyncpg driver prefix; psycopg2 needs plain."""
    return url.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")


SYNC_URL = _to_sync_dsn(TEST_DB_URL)


def _db_reachable(url: str) -> bool:
    """Cheap TCP probe, same pattern as tests/api/test_admin_users_db.py."""
    if not url:
        return False
    try:
        parts = urllib.parse.urlsplit(url)
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not TEST_DB_URL,
        reason="TEST_DATABASE_URL is not set; refusing to fall back to the live DATABASE_URL",
    ),
    pytest.mark.skipif(
        bool(TEST_DB_URL) and not _db_reachable(TEST_DB_URL),
        reason="TEST_DATABASE_URL is unreachable",
    ),
]


def _raw_connect() -> psycopg2.extensions.connection:
    """Fresh autocommit psycopg2 connection to the test DB (the committed view)."""
    conn = psycopg2.connect(SYNC_URL)
    conn.autocommit = True
    return conn


def _regclass(cur: psycopg2.extensions.cursor, name: str) -> str | None:
    cur.execute("SELECT to_regclass(%s)", (name,))
    return cur.fetchone()[0]


def _ensure_schema(raw: psycopg2.extensions.connection) -> dict[str, bool]:
    """Create chat tables from the migration DDL when missing; never drop pre-existing ones."""
    created = {"users": False, "chat_threads": False, "chat_messages": False}
    cur = raw.cursor()
    try:
        if _regclass(cur, "public.users") is None:
            cur.execute(_USERS_STUB_DDL)
            created["users"] = True
        if _regclass(cur, "public.chat_threads") is None:
            cur.execute(_CHAT_THREADS_DDL)
            created["chat_threads"] = True
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ct_user ON chat_threads(user_id, created_at)")
        if _regclass(cur, "public.chat_messages") is None:
            cur.execute(_CHAT_MESSAGES_DDL)
            created["chat_messages"] = True
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_thread_seq ON chat_messages(thread_id, seq)")
    finally:
        cur.close()
    return created


def _seed_guard(raw: psycopg2.extensions.connection) -> dict[str, Any]:
    """Seed a committed foreign thread + 2 messages (live-data stand-in).

    user_id is picked from an existing users row when the test DB has one
    (a migrated DB), so the guard emulates "another user's thread".
    """
    key = f"live-guard-{uuid.uuid4().hex[:12]}"
    cur = raw.cursor()
    try:
        cur.execute(
            "INSERT INTO chat_threads (user_id, context_type, context_key, title) "
            "VALUES ((SELECT id FROM users LIMIT 1), 'teacher', %s, %s) RETURNING id",
            (key, _GUARD_TITLE),
        )
        thread_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO chat_messages (thread_id, role, seq, blocks_json) "
            "VALUES (%s, 'user', 1, %s), (%s, 'assistant', 2, %s)",
            (thread_id, '{"text": "live message 1"}', thread_id, '{"text": "live message 2"}'),
        )
    finally:
        cur.close()
    return {"id": thread_id, "key": key, "msg_count": 2}


def _assert_guard_intact(raw: psycopg2.extensions.connection, guard: dict[str, Any]) -> None:
    """Tripwire: the foreign thread must still exist after the whole session."""
    cur = raw.cursor()
    try:
        for table in ("chat_threads", "chat_messages"):
            assert _regclass(cur, f"public.{table}") is not None, f"table {table} was dropped by the test session"
        cur.execute("SELECT COUNT(*) FROM chat_threads WHERE id = %s", (guard["id"],))
        n = cur.fetchone()[0]
        assert n == 1, f"foreign thread {guard['key']} did not survive the suite (count={n})"
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = %s", (guard["id"],))
        n = cur.fetchone()[0]
        expected = guard["msg_count"]
        assert n == expected, f"foreign messages of {guard['key']} did not survive (count={n})"
    finally:
        cur.close()


def _scoped_cleanup(raw: psycopg2.extensions.connection) -> None:
    """Delete only this run's tagged rows (belt-and-suspenders after the rollback)."""
    cur = raw.cursor()
    try:
        cur.execute(
            "DELETE FROM chat_messages WHERE thread_id IN (SELECT id FROM chat_threads WHERE context_key LIKE %s)",
            (f"{TAG}-%",),
        )
        cur.execute("DELETE FROM chat_threads WHERE context_key LIKE %s", (f"{TAG}-%",))
    finally:
        cur.close()


def _teardown_test_db(raw: psycopg2.extensions.connection, guard: dict[str, Any], created: dict[str, bool]) -> None:
    """Always-attempted cleanup: remove guard rows, drop only tables we created.

    Guarded by to_regclass checks so a broken test that dropped a table cannot
    mask the tripwire error with a confusing UndefinedTable failure here.
    """
    cur = raw.cursor()
    try:
        if _regclass(cur, "public.chat_threads") is not None:
            if _regclass(cur, "public.chat_messages") is not None:
                cur.execute("DELETE FROM chat_messages WHERE thread_id = %s", (guard["id"],))
            cur.execute("DELETE FROM chat_threads WHERE id = %s", (guard["id"],))
        for table in ("chat_messages", "chat_threads", "users"):
            if created[table]:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        cur.close()
    raw.close()


@pytest.fixture(scope="session")
def _chat_env() -> Iterator[dict[str, Any]]:
    """Test-DB bootstrap: schema from migration DDL when missing + foreign guard rows."""
    try:
        raw = _raw_connect()
    except psycopg2.Error as e:
        pytest.skip(f"cannot connect to TEST_DATABASE_URL: {e}")
    created = _ensure_schema(raw)
    guard = _seed_guard(raw)
    try:
        yield {"raw": raw, "guard": guard, "created": created}
    finally:
        try:
            # Tripwire: fail loudly when the suite destroyed live-like rows.
            _assert_guard_intact(raw, guard)
        finally:
            # Cleanup runs even when the tripwire fires (and never masks it).
            _teardown_test_db(raw, guard, created)


@pytest.fixture(scope="session")
def _engine(_chat_env) -> Iterator[Any]:
    """NullPool engine: every test gets its own short-lived connection."""
    engine = create_engine(SYNC_URL, poolclass=NullPool)
    yield engine
    engine.dispose()


@pytest.fixture()
def sess(_chat_env, _engine) -> Iterator[Session]:
    """Per-test ORM session inside a savepoint of an outer transaction.

    ``join_transaction_mode="create_savepoint"`` makes ``sess.commit()`` release
    only a savepoint, so even committed test data dies with the outer rollback.
    """
    conn = _engine.connect()
    outer = conn.begin()
    s = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield s
    finally:
        try:
            s.rollback()
            s.close()
        finally:
            try:
                outer.rollback()
            finally:
                conn.close()
        # The rollback already handled the fast path; this catches anything a
        # driver corner case might have persisted - scoped by the unique tag.
        _scoped_cleanup(_chat_env["raw"])


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_thread(sess: Session, **kw) -> ChatThread:
    """Every thread gets a context_key tagged with this run's unique tag."""
    defaults = {
        "context_type": "teacher",
        "context_key": f"{TAG}-{next(_THREAD_SEQ):03d}",
    }
    defaults.update(kw)
    t = ChatThread(**defaults)
    sess.add(t)
    sess.flush()
    return t


def _make_msg(sess: Session, thread: ChatThread, role: str, seq: int, **kw):
    m = ChatMessage(thread=thread, role=role, seq=seq, **kw)
    sess.add(m)
    sess.flush()
    return m


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_foreign_thread_survives(sess: Session, _chat_env) -> None:
    """Isolation proof: the foreign thread survives, test rows are never committed."""
    guard = _chat_env["guard"]

    # Our own activity: create + "commit" a thread the way every CRUD test does.
    t = _make_thread(sess, title="Isolation probe")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "must never be committed"})
    sess.commit()

    # Committed view via a second connection: guard intact, no test rows leaked.
    raw = _raw_connect()
    try:
        cur = raw.cursor()
        cur.execute("SELECT COUNT(*) FROM chat_threads WHERE context_key = %s", (guard["key"],))
        assert cur.fetchone()[0] == 1, "foreign thread was deleted"
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = %s", (guard["id"],))
        assert cur.fetchone()[0] == guard["msg_count"], "foreign messages were deleted"
        cur.execute("SELECT COUNT(*) FROM chat_threads WHERE context_key LIKE %s", (f"{TAG}-%",))
        assert cur.fetchone()[0] == 0, "test rows leaked outside the test transaction"
        cur.close()
    finally:
        raw.close()

    # Same-transaction view: the foreign thread and its messages are readable.
    row = sess.execute(select(ChatThread).where(ChatThread.context_key == guard["key"])).scalar_one()
    assert row.title == _GUARD_TITLE
    assert len(row.messages) == guard["msg_count"]


def test_create_thread_and_messages(sess: Session) -> None:
    """Basic creation: thread + 2 messages (user + assistant)."""
    t = _make_thread(sess, title="Linear Algebra")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "What is a matrix?"})
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "A matrix is ..."})
    sess.commit()

    row = sess.execute(select(ChatThread).where(ChatThread.id == t.id)).scalar_one()
    assert row.title == "Linear Algebra"
    assert len(row.messages) == 2


def test_seq_uniqueness_per_thread(sess: Session) -> None:
    """Duplicate (thread_id, seq) must fail UNIQUE constraint."""
    t = _make_thread(sess)
    _make_msg(sess, t, "user", 1)
    sess.flush()
    with pytest.raises(Exception):  # psycopg2.errors.UniqueViolation
        m = ChatMessage(thread=t, role="assistant", seq=1)
        sess.add(m)
        sess.flush()


def test_query_by_thread_seq(sess: Session) -> None:
    """SELECT WHERE (thread_id, seq) — index: composite idx_cm_thread_seq."""
    t = _make_thread(sess, title="Calculus")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "hello"})
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "hi"})
    _make_msg(sess, t, "user", 3, blocks_json={"text": "bye"})
    sess.commit()

    msg = sess.execute(select(ChatMessage).where(ChatMessage.thread_id == t.id, ChatMessage.seq == 2)).scalar_one()
    assert msg.role == "assistant"
    assert msg.blocks_json == {"text": "hi"}

    rows = (
        sess.execute(
            select(ChatMessage).where(ChatMessage.thread_id == t.id, ChatMessage.seq >= 2).order_by(ChatMessage.seq)
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2


def test_cascade_delete_thread_removes_messages(sess: Session) -> None:
    """Deleting a thread CASCADE-deletes all its messages."""
    t = _make_thread(sess, title="Delete me")
    _make_msg(sess, t, "user", 1)
    _make_msg(sess, t, "assistant", 2)
    sess.commit()

    cnt = sess.execute(select(ChatMessage).where(ChatMessage.thread_id == t.id)).scalars().all()
    assert len(cnt) == 2

    sess.delete(t)
    sess.commit()

    cnt = sess.execute(select(ChatMessage).where(ChatMessage.thread_id == t.id)).scalars().all()
    assert len(cnt) == 0


def test_context_type_check_constraint(sess: Session) -> None:
    """Only 'teacher' and 'student' are allowed."""
    t = _make_thread(sess)
    sess.flush()  # OK = 'teacher'
    with pytest.raises(Exception):
        t.context_type = "boss"
        sess.flush()


def test_role_check_constraint(sess: Session) -> None:
    """Only 'user' and 'assistant' are allowed."""
    t = _make_thread(sess)
    _make_msg(sess, t, "user", 1)
    sess.flush()
    with pytest.raises(Exception):
        m = _make_msg(sess, t, "user", 99)
        m.role = "system"
        sess.flush()


def test_thread_user_set_null(sess: Session) -> None:
    """user_id FK: SET NULL on user delete is modelled in the schema."""
    t = _make_thread(sess)
    sess.flush()
    assert t.user_id is None


def test_messages_order_by_seq(sess: Session) -> None:
    """Messages can be retrieved in chronological order via seq ASC."""
    t = _make_thread(sess, title="Ordering")
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "b"})
    _make_msg(sess, t, "user", 1, blocks_json={"text": "a"})
    _make_msg(sess, t, "assistant", 3, blocks_json={"text": "c"})
    sess.commit()

    ordered = (
        sess.execute(select(ChatMessage).where(ChatMessage.thread_id == t.id).order_by(ChatMessage.seq)).scalars().all()
    )
    assert [m.blocks_json["text"] for m in ordered] == ["a", "b", "c"]


def test_model_used_null_on_message(sess: Session) -> None:
    """model_used is nullable — a user message may not have it."""
    t = _make_thread(sess)
    msg = _make_msg(sess, t, "user", 1, model_used=None)
    sess.flush()
    assert msg.model_used is None

    msg2 = _make_msg(sess, t, "assistant", 2, model_used="deepseek-ai/DeepSeek-V4.1-Pro-0813")
    sess.flush()
    assert msg2.model_used == "deepseek-ai/DeepSeek-V4.1-Pro-0813"
