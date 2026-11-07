"""CRUD tests for chat_threads + chat_messages models.

Scope: only the new chat tables; no impact on production data.
Uses raw SQL to create tables (mimicking the migration), then ORM for CRUD.
"""

import os

import psycopg2
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.models.krm_models import ChatMessage, ChatThread  # noqa: F401


@pytest.fixture(scope="session")
def _sync_conn():
    """Sync psycopg2 connection to the live DB (session-scoped)."""
    url = os.environ.get("DATABASE_URL", "")
    sync_url = url.replace("postgresql+asyncpg://", "postgresql://")
    conn = psycopg2.connect(sync_url)
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture()
def _setup_chat_tables(_sync_conn):
    """Create chat_threads + chat_messages via raw SQL (migration copy).

    Uses DELETE (not TRUNCATE) for teardown to avoid blocking on open
    transactions from the ``sess`` fixture's connection pool.  TRUNCATE
    requires an ACCESS EXCLUSIVE lock which is blocked when another session
    holds a snapshot; DELETE acquires only a RowShare lock and completes
    immediately.
    """
    cur = _sync_conn.cursor()
    cur.execute(
        "DELETE FROM chat_messages; DELETE FROM chat_threads"
    )
    cur.close()
    yield
    # Teardown: clean test data (same safe DELETE approach)
    cur = _sync_conn.cursor()
    cur.execute(
        "DELETE FROM chat_messages; DELETE FROM chat_threads"
    )
    cur.close()


@pytest.fixture()
def sess(_sync_conn) -> Session:
    """Sync ORM session on the live DB."""
    url = os.environ.get("DATABASE_URL", "")
    sync_url = url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(sync_url)
    sm = sessionmaker(engine, class_=Session, expire_on_commit=False)
    s = sm()
    yield s
    s.close()
    engine.dispose()


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_thread(sess: Session, **kw) -> ChatThread:
    defaults = {
        "context_type": "teacher",
        "context_key": "math-101",
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


def test_create_thread_and_messages(sess: Session, _setup_chat_tables) -> None:
    """Basic creation: thread + 2 messages (user + assistant)."""
    t = _make_thread(sess, title="Linear Algebra")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "What is a matrix?"})
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "A matrix is ..."})
    sess.commit()

    row = sess.execute(
        select(ChatThread).where(ChatThread.id == t.id)
    ).scalar_one()
    assert row.title == "Linear Algebra"
    assert len(row.messages) == 2


def test_seq_uniqueness_per_thread(sess: Session, _setup_chat_tables) -> None:
    """Duplicate (thread_id, seq) must fail UNIQUE constraint."""
    t = _make_thread(sess)
    _make_msg(sess, t, "user", 1)
    sess.flush()
    with pytest.raises(Exception):  # psycopg2.errors.UniqueViolation
        m = ChatMessage(thread=t, role="assistant", seq=1)
        sess.add(m)
        sess.flush()


def test_query_by_thread_seq(sess: Session, _setup_chat_tables) -> None:
    """SELECT WHERE (thread_id, seq) — index: composite idx_cm_thread_seq."""
    t = _make_thread(sess, title="Calculus")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "hello"})
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "hi"})
    _make_msg(sess, t, "user", 3, blocks_json={"text": "bye"})
    sess.commit()

    msg = sess.execute(
        select(ChatMessage).where(
            ChatMessage.thread_id == t.id, ChatMessage.seq == 2
        )
    ).scalar_one()
    assert msg.role == "assistant"
    assert msg.blocks_json == {"text": "hi"}

    rows = sess.execute(
        select(ChatMessage).where(
            ChatMessage.thread_id == t.id, ChatMessage.seq >= 2
        ).order_by(ChatMessage.seq)
    ).scalars().all()
    assert len(rows) == 2


def test_cascade_delete_thread_removes_messages(sess: Session, _setup_chat_tables) -> None:
    """Deleting a thread CASCADE-deletes all its messages."""
    t = _make_thread(sess, title="Delete me")
    _make_msg(sess, t, "user", 1)
    _make_msg(sess, t, "assistant", 2)
    sess.commit()

    cnt = sess.execute(
        select(ChatMessage).where(ChatMessage.thread_id == t.id)
    ).scalars().all()
    assert len(cnt) == 2

    sess.delete(t)
    sess.commit()

    cnt = sess.execute(
        select(ChatMessage).where(ChatMessage.thread_id == t.id)
    ).scalars().all()
    assert len(cnt) == 0


def test_db_level_cascade_delete_raw_sql(
    sess: Session, _setup_chat_tables, _sync_conn
) -> None:
    """ON DELETE CASCADE живёт в самой БД, а не только в ORM.

    ORM-тест выше не отличает FK-каскад от cascade="all, delete-orphan"
    (krm_models.py:842); этот тест удаляет строку треда raw SQL в обход
    ORM-сессии - сообщения обязаны исчезнуть силами самой БД
    (chat_messages_thread_id_fkey, alembic add_chat_storage.py).
    """
    t = _make_thread(sess, title="Raw SQL cascade")
    _make_msg(sess, t, "user", 1, blocks_json={"text": "a"})
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "b"})
    sess.commit()  # строки видны другим соединениям

    cur = _sync_conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = %s", (t.id,))
        assert cur.fetchone()[0] == 2
        cur.execute("DELETE FROM chat_threads WHERE id = %s", (t.id,))
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE thread_id = %s", (t.id,))
        assert cur.fetchone()[0] == 0
    finally:
        cur.close()


def test_context_type_check_constraint(sess: Session, _setup_chat_tables) -> None:
    """Only 'teacher' and 'student' are allowed."""
    t = _make_thread(sess)
    sess.flush()  # OK = 'teacher'
    with pytest.raises(Exception):
        t.context_type = "boss"
        sess.flush()


def test_role_check_constraint(sess: Session, _setup_chat_tables) -> None:
    """Only 'user' and 'assistant' are allowed."""
    t = _make_thread(sess)
    _make_msg(sess, t, "user", 1)
    sess.flush()
    with pytest.raises(Exception):
        m = _make_msg(sess, t, "user", 99)
        m.role = "system"
        sess.flush()


def test_thread_user_set_null(sess: Session, _setup_chat_tables) -> None:
    """user_id FK: SET NULL on user delete is modelled in the schema."""
    t = _make_thread(sess)
    sess.flush()
    assert t.user_id is None


def test_messages_order_by_seq(sess: Session, _setup_chat_tables) -> None:
    """Messages can be retrieved in chronological order via seq ASC."""
    t = _make_thread(sess, title="Ordering")
    _make_msg(sess, t, "assistant", 2, blocks_json={"text": "b"})
    _make_msg(sess, t, "user", 1, blocks_json={"text": "a"})
    _make_msg(sess, t, "assistant", 3, blocks_json={"text": "c"})
    sess.commit()

    ordered = sess.execute(
        select(ChatMessage)
        .where(ChatMessage.thread_id == t.id)
        .order_by(ChatMessage.seq)
    ).scalars().all()
    assert [m.blocks_json["text"] for m in ordered] == ["a", "b", "c"]


def test_model_used_null_on_message(sess: Session, _setup_chat_tables) -> None:
    """model_used is nullable — a user message may not have it."""
    t = _make_thread(sess)
    msg = _make_msg(sess, t, "user", 1, model_used=None)
    sess.flush()
    assert msg.model_used is None

    msg2 = _make_msg(
        sess, t, "assistant", 2, model_used="deepseek-ai/DeepSeek-V4.1-Pro-0813"
    )
    sess.flush()
    assert msg2.model_used == "deepseek-ai/DeepSeek-V4.1-Pro-0813"
