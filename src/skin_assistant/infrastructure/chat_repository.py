"""
Persist chat messages to MySQL (skinme_db). Uses same MYSQL_* env as SkinMeDBClient.
Creates chat_sessions and chat_messages tables if they do not exist.
"""
from typing import List, Optional

from skin_assistant.config import get_settings


def _get_connection():
    """Return a DB connection or None if MySQL not configured."""
    s = get_settings()
    if not getattr(s, "use_mysql_db", None) and not (
        getattr(s, "mysql_host", None) and getattr(s, "mysql_user", None) and getattr(s, "mysql_database", None)
    ):
        # Recheck use_mysql_db style
        if not (s.mysql_host and s.mysql_user and s.mysql_database):
            return None
    try:
        import pymysql
    except ImportError:
        return None
    try:
        return pymysql.connect(
            host=s.mysql_host,
            port=getattr(s, "mysql_port", 3306),
            user=s.mysql_user,
            password=s.mysql_password,
            database=s.mysql_database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
    except Exception:
        return None


def _ensure_tables(conn) -> bool:
    """Create chat_sessions and chat_messages if not exist. Returns True on success."""
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL UNIQUE,
                    user_id VARCHAR(128) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session (session_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    image_analysis VARCHAR(255) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_created (session_id, created_at)
                )
            """)
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


class ChatRepository:
    """Save and load chat messages in MySQL (skinme_db)."""

    def __init__(self):
        self._settings = get_settings()

    def is_available(self) -> bool:
        return bool(self._settings.use_mysql_db)

    def _conn(self):
        return _get_connection()

    def ensure_tables(self) -> bool:
        conn = self._conn()
        if conn is None:
            return False
        try:
            return _ensure_tables(conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def ensure_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Insert session if not exists. Returns True on success."""
        conn = self._conn()
        if conn is None:
            return False
        try:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO chat_sessions (session_id, user_id) VALUES (%s, %s)",
                    (session_id[:128], (user_id or "")[:128] or None),
                )
            conn.commit()
            return True
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        image_analysis: Optional[str] = None,
    ) -> bool:
        """Append one chat message. Creates session if needed. Returns True on success."""
        if not self.is_available():
            return False
        self.ensure_session(session_id)
        conn = self._conn()
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (session_id, role, content, image_analysis) VALUES (%s, %s, %s, %s)",
                    (session_id[:128], role[:32], content[:65535], (image_analysis or "")[:255] or None),
                )
            conn.commit()
            return True
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_history(self, session_id: str, limit: int = 100) -> List[dict]:
        """Return messages for session, newest last. Each dict: role, content, image_analysis, created_at."""
        conn = self._conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content, image_analysis, created_at FROM chat_messages WHERE session_id = %s ORDER BY id ASC LIMIT %s",
                    (session_id[:128], limit),
                )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass
