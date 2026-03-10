"""
database/connection.py
-----------------------
Database connection manager supporting:
  - MySQL / MariaDB (production) via mysql-connector-python
  - SQLite in-memory (development / CI / testing) as automatic fallback

The dual-backend design means the entire pipeline can be developed,
tested, and CI'd without a MySQL server.  In production, point the
config at your MySQL host and all the same queries run identically.

Design pattern: Repository / Unit of Work
  - DatabaseManager handles connections and transactions
  - All queries are centralised in repositories (see repositories.py)
  - No raw SQL leaks into business logic
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from privacy_pipeline.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_MYSQL_AVAILABLE = False
try:
    import mysql.connector  # type: ignore
    from mysql.connector import pooling as mysql_pooling  # type: ignore
    _MYSQL_AVAILABLE = True
except ImportError:
    logger.debug("mysql-connector-python not installed — SQLite fallback available")


class DatabaseError(Exception):
    """Raised when a database operation fails after retries."""


class ConnectionConfig:
    """Validated database connection parameters."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "privacy_pipeline_db",
        user: str = "pp_user",
        password: str = "",
        charset: str = "utf8mb4",
        pool_size: int = 5,
        connect_timeout: int = 10,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        fallback_to_sqlite: bool = True,
        sqlite_path: str = ":memory:",
    ) -> None:
        self.host              = host
        self.port              = port
        self.database          = database
        self.user              = user
        self.password          = password
        self.charset           = charset
        self.pool_size         = pool_size
        self.connect_timeout   = connect_timeout
        self.max_retries       = max_retries
        self.retry_delay_sec   = retry_delay_sec
        self.fallback_to_sqlite = fallback_to_sqlite
        self.sqlite_path       = sqlite_path

    @classmethod
    def from_config(cls, cfg) -> "ConnectionConfig":
        """Build from a Config object loaded from YAML."""
        db = cfg.database
        return cls(
            host              = db.get("host", "localhost"),
            port              = int(db.get("port", 3306)),
            database          = db.get("name", "privacy_pipeline_db"),
            user              = db.get("user", "pp_user"),
            password          = db.get("password", ""),
            charset           = db.get("charset", "utf8mb4"),
            pool_size         = int(db.get("pool_size", 5)),
            connect_timeout   = int(db.get("connect_timeout", 10)),
            max_retries       = int(db.get("max_retries", 3)),
            retry_delay_sec   = float(db.get("retry_delay_sec", 2.0)),
            fallback_to_sqlite= bool(db.get("fallback_to_sqlite", True)),
            sqlite_path       = db.get("sqlite_path", ":memory:"),
        )


# ---------------------------------------------------------------------------
# SQLite backend (development / CI)
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    """Thread-safe SQLite backend using per-thread connections."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path  = path
        self._local = threading.local()
        logger.info("Database backend: SQLite (%s)", path)

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self._path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; explicit BEGIN used
            )
            conn.row_factory = sqlite3.Row
            # SQLite pragma for WAL mode (better concurrency)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def execute(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        return self._get_conn().execute(sql, params)

    def executemany(self, sql: str, data: List[Tuple]) -> sqlite3.Cursor:
        return self._get_conn().executemany(sql, data)

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Dict]:
        cur = self.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Dict]:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # Translate MySQL DDL → SQLite-compatible DDL
    @staticmethod
    def adapt_sql(sql: str) -> str:
        """
        Convert MySQL-flavoured SQL to SQLite syntax.
        Handles the small set of differences used in our schema.
        """
        replacements = [
            ("ENGINE=InnoDB",          ""),
            ("DEFAULT CHARSET=utf8mb4",""),
            ("COLLATE utf8mb4_unicode_ci", ""),
            ("AUTO_INCREMENT",         "AUTOINCREMENT"),
            ("TINYINT(1)",             "INTEGER"),
            ("DATETIME(6)",            "DATETIME"),
            ("MEDIUMTEXT",             "TEXT"),
            ("JSON",                   "TEXT"),
            ("`",                      ""),    # backtick quoting not needed in SQLite
        ]
        adapted = sql
        for old, new in replacements:
            adapted = adapted.replace(old, new)
        return adapted


# ---------------------------------------------------------------------------
# MySQL backend (production)
# ---------------------------------------------------------------------------

class _MySQLBackend:
    """
    MySQL backend using a connection pool.
    Requires: pip install mysql-connector-python
    """

    def __init__(self, cfg: ConnectionConfig) -> None:
        pool_cfg = {
            "pool_name":       "ppvp_pool",
            "pool_size":       cfg.pool_size,
            "host":            cfg.host,
            "port":            cfg.port,
            "database":        cfg.database,
            "user":            cfg.user,
            "password":        cfg.password,
            "charset":         cfg.charset,
            "connection_timeout": cfg.connect_timeout,
            "use_unicode":     True,
            "autocommit":      False,
        }
        self._pool = mysql_pooling.MySQLConnectionPool(**pool_cfg)
        logger.info(
            "Database backend: MySQL @ %s:%d/%s (pool_size=%d)",
            cfg.host, cfg.port, cfg.database, cfg.pool_size,
        )

    @contextmanager
    def transaction(self):
        conn = self._pool.get_connection()
        try:
            conn.start_transaction()
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Tuple = ()):
        with self._pool.get_connection() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params)
            conn.commit()
            return cur

    def executemany(self, sql: str, data: List[Tuple]):
        with self._pool.get_connection() as conn:
            cur = conn.cursor(dictionary=True)
            cur.executemany(sql, data)
            conn.commit()
            return cur

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Dict]:
        conn = self._pool.get_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params)
            return cur.fetchall() or []
        finally:
            conn.close()

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Dict]:
        conn = self._pool.get_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()

    def close(self) -> None:
        pass  # Pool manages its own lifecycle

    @staticmethod
    def adapt_sql(sql: str) -> str:
        return sql  # MySQL DDL is already MySQL-flavoured


# ---------------------------------------------------------------------------
# DatabaseManager — public interface
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Unified database manager.  Automatically selects MySQL or SQLite backend.

    Usage
    -----
    >>> db = DatabaseManager.from_config(cfg)
    >>> db.initialize_schema()
    >>> rows = db.fetchall("SELECT * FROM detection_events LIMIT 10")
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, force_sqlite: bool = False) -> "DatabaseManager":
        """
        Create a DatabaseManager from a Config object.

        Parameters
        ----------
        cfg:           Config loaded from YAML.
        force_sqlite:  Skip MySQL attempt and use SQLite directly (for tests).
        """
        conn_cfg = ConnectionConfig.from_config(cfg)
        return cls._create(conn_cfg, force_sqlite=force_sqlite)

    @classmethod
    def for_testing(cls) -> "DatabaseManager":
        """Create an in-memory SQLite instance for unit tests."""
        backend = _SQLiteBackend(":memory:")
        instance = cls(backend)
        instance.initialize_schema()
        return instance

    @classmethod
    def _create(
        cls, conn_cfg: ConnectionConfig, force_sqlite: bool = False
    ) -> "DatabaseManager":
        if not force_sqlite and _MYSQL_AVAILABLE:
            # Attempt MySQL connection with retries
            for attempt in range(1, conn_cfg.max_retries + 1):
                try:
                    backend = _MySQLBackend(conn_cfg)
                    logger.info("MySQL connection established (attempt %d)", attempt)
                    return cls(backend)
                except Exception as exc:
                    logger.warning(
                        "MySQL connection attempt %d/%d failed: %s",
                        attempt, conn_cfg.max_retries, exc,
                    )
                    if attempt < conn_cfg.max_retries:
                        time.sleep(conn_cfg.retry_delay_sec)

            if not conn_cfg.fallback_to_sqlite:
                raise DatabaseError(
                    f"Could not connect to MySQL after {conn_cfg.max_retries} attempts"
                )
            logger.warning("Falling back to SQLite backend")

        # SQLite path
        backend = _SQLiteBackend(conn_cfg.sqlite_path)
        return cls(backend)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create all tables if they do not exist."""
        from privacy_pipeline.database.schema import SCHEMA_DDL
        for ddl in SCHEMA_DDL:
            adapted = self._backend.adapt_sql(ddl)
            self._backend.execute(adapted)
        logger.info("Database schema initialised")

    def is_mysql(self) -> bool:
        return isinstance(self._backend, _MySQLBackend)

    def is_sqlite(self) -> bool:
        return isinstance(self._backend, _SQLiteBackend)

    # ------------------------------------------------------------------
    # Delegate query methods
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self):
        with self._backend.transaction() as conn:
            yield conn

    def execute(self, sql: str, params: Tuple = ()):
        return self._backend.execute(sql, params)

    def executemany(self, sql: str, data: List[Tuple]):
        return self._backend.executemany(sql, data)

    def fetchall(self, sql: str, params: Tuple = ()) -> List[Dict]:
        return self._backend.fetchall(sql, params)

    def fetchone(self, sql: str, params: Tuple = ()) -> Optional[Dict]:
        return self._backend.fetchone(sql, params)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
