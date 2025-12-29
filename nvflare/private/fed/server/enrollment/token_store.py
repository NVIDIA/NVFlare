# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token State Store implementations.

This module provides storage backends for tracking token usage,
revocation status, and enrolled entities.

Available implementations:
- SQLiteTokenStateStore: SQLite database (default, development/single server)
- PostgresTokenStateStore: PostgreSQL database (production/distributed)
  Located in nvflare.app_opt.enrollment.postgres_token_store (requires psycopg2)
"""

import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List

logger = logging.getLogger(__name__)


class TokenStateStore(ABC):
    """Abstract base class for token state storage.
    
    Implementations must be thread-safe and handle concurrent access.
    """

    @abstractmethod
    def record_use(self, token_id: str, enrolled_entity: str) -> int:
        """Record token usage and return current use count.
        
        Args:
            token_id: Unique token identifier (JWT jti claim)
            enrolled_entity: Name of the entity being enrolled
            
        Returns:
            Current usage count after recording
        """
        pass

    @abstractmethod
    def get_use_count(self, token_id: str) -> int:
        """Get current usage count for a token.
        
        Args:
            token_id: Unique token identifier
            
        Returns:
            Number of times the token has been used
        """
        pass

    @abstractmethod
    def is_revoked(self, token_id: str) -> bool:
        """Check if token has been revoked.
        
        Args:
            token_id: Unique token identifier
            
        Returns:
            True if token is revoked
        """
        pass

    @abstractmethod
    def revoke(self, token_id: str) -> None:
        """Revoke a token.
        
        Args:
            token_id: Unique token identifier
        """
        pass

    @abstractmethod
    def get_enrolled_entities(self, token_id: str) -> List[str]:
        """Get list of entities enrolled with this token.
        
        Args:
            token_id: Unique token identifier
            
        Returns:
            List of entity names enrolled with this token
        """
        pass

    def close(self) -> None:
        """Close any open connections. Override if needed."""
        pass


# =============================================================================
# SQLite Implementation (Default - Development / Single Server)
# =============================================================================

class SQLiteTokenStateStore(TokenStateStore):
    """SQLite-based token state storage.
    
    This is the default storage backend.
    
    Suitable for:
    - Development environments
    - Single-server production
    - Good concurrency with WAL mode
    
    Features:
    - ACID transactions
    - Thread-safe with connection per thread
    - Automatic schema creation
    - WAL mode for better concurrency
    """

    def __init__(self, db_path: str = "token_state.db"):
        """Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tokens (
                token_id TEXT PRIMARY KEY,
                uses INTEGER DEFAULT 0,
                revoked INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (token_id) REFERENCES tokens(token_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_enrollments_token_id 
                ON enrollments(token_id);
        """)
        conn.commit()

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def record_use(self, token_id: str, enrolled_entity: str) -> int:
        with self._transaction() as conn:
            # Upsert token record
            conn.execute("""
                INSERT INTO tokens (token_id, uses, revoked)
                VALUES (?, 1, 0)
                ON CONFLICT(token_id) DO UPDATE SET
                    uses = uses + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (token_id,))
            
            # Record enrollment
            conn.execute("""
                INSERT INTO enrollments (token_id, entity)
                VALUES (?, ?)
            """, (token_id, enrolled_entity))
            
            # Get current count
            cursor = conn.execute(
                "SELECT uses FROM tokens WHERE token_id = ?",
                (token_id,)
            )
            row = cursor.fetchone()
            return row["uses"] if row else 1

    def get_use_count(self, token_id: str) -> int:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT uses FROM tokens WHERE token_id = ?",
            (token_id,)
        )
        row = cursor.fetchone()
        return row["uses"] if row else 0

    def is_revoked(self, token_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT revoked FROM tokens WHERE token_id = ?",
            (token_id,)
        )
        row = cursor.fetchone()
        return bool(row["revoked"]) if row else False

    def revoke(self, token_id: str) -> None:
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO tokens (token_id, uses, revoked)
                VALUES (?, 0, 1)
                ON CONFLICT(token_id) DO UPDATE SET
                    revoked = 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (token_id,))

    def get_enrolled_entities(self, token_id: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT entity FROM enrollments WHERE token_id = ? ORDER BY enrolled_at",
            (token_id,)
        )
        return [row["entity"] for row in cursor.fetchall()]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# Factory Function
# =============================================================================

def create_token_store(
    backend: str = "sqlite",
    **kwargs
) -> TokenStateStore:
    """Factory function to create token state store.
    
    Args:
        backend: Storage backend type: "sqlite" (default) or "postgres"
        **kwargs: Backend-specific configuration
        
    Returns:
        TokenStateStore instance
        
    Examples:
        # SQLite (default, development)
        store = create_token_store("sqlite", db_path="/path/to/tokens.db")
        
        # PostgreSQL (production) - requires psycopg2
        store = create_token_store(
            "postgres",
            host="db.example.com",
            database="nvflare",
            user="nvflare",
            password="secret"
        )
    """
    backend_lower = backend.lower()
    
    if backend_lower == "sqlite":
        return SQLiteTokenStateStore(**kwargs)
    
    elif backend_lower == "postgres":
        try:
            from nvflare.app_opt.enrollment.postgres_token_store import PostgresTokenStateStore
            return PostgresTokenStateStore(**kwargs)
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires psycopg (v3) or psycopg2. "
                "Install with: pip install psycopg[binary] or pip install psycopg2-binary. "
                f"Original error: {e}"
            )
    
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: sqlite, postgres"
        )
