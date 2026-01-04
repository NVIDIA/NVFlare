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

"""Enrollment Store - Tracks enrolled entities and pending requests.

This module provides storage backends for the Certificate Service to track:
- Enrolled entities (sites and users that have completed enrollment)
- Pending requests (enrollment requests awaiting manual approval)

Entity uniqueness is determined by (name, entity_type) pair.
"""

import fnmatch
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class EnrolledEntity:
    """An enrolled site or user."""

    name: str  # Entity name (site-1, admin@org.com)
    entity_type: str  # client | relay | server | admin
    enrolled_at: datetime
    org: Optional[str] = None  # Organization
    role: Optional[str] = None  # Role (for admin tokens only)


@dataclass
class PendingRequest:
    """Pending enrollment request awaiting admin approval."""

    name: str  # Entity name
    entity_type: str  # client | relay | server | admin
    org: str
    csr_pem: str
    submitted_at: datetime
    expires_at: datetime
    token_subject: str
    role: Optional[str] = None  # Role (for admin tokens only)
    source_ip: Optional[str] = None
    signed_cert: Optional[str] = None
    approved: bool = False
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None


class EnrollmentStore(ABC):
    """Abstract interface for enrollment state storage.

    Tracks enrolled entities (sites and users) and pending requests.
    Entity uniqueness is determined by (name, entity_type) pair.
    """

    # ─────────────────────────────────────────────────────
    # Enrolled Entities (Sites and Users)
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def is_enrolled(self, name: str, entity_type: str) -> bool:
        """Check if an entity is already enrolled."""
        pass

    @abstractmethod
    def add_enrolled(self, entity: EnrolledEntity) -> None:
        """Mark an entity as enrolled. Also removes from pending."""
        pass

    @abstractmethod
    def get_enrolled(self, entity_type: Optional[str] = None) -> List[EnrolledEntity]:
        """Get enrolled entities, optionally filtered by type."""
        pass

    # ─────────────────────────────────────────────────────
    # Pending Requests
    # ─────────────────────────────────────────────────────

    @abstractmethod
    def is_pending(self, name: str, entity_type: str) -> bool:
        """Check if an entity has a pending request."""
        pass

    @abstractmethod
    def add_pending(self, request: PendingRequest) -> None:
        """Add a new pending request."""
        pass

    @abstractmethod
    def get_pending(self, name: str, entity_type: str) -> Optional[PendingRequest]:
        """Get pending request for an entity."""
        pass

    @abstractmethod
    def get_all_pending(self, entity_type: Optional[str] = None) -> List[PendingRequest]:
        """Get all pending requests, optionally filtered by type."""
        pass

    @abstractmethod
    def approve_pending(
        self,
        name: str,
        entity_type: str,
        signed_cert: str,
        approved_by: str,
    ) -> Optional[PendingRequest]:
        """Approve a pending request and store signed certificate.

        Returns the approved request if found, None otherwise.
        """
        pass

    @abstractmethod
    def reject_pending(self, name: str, entity_type: str, reason: str) -> bool:
        """Reject and remove a pending request.

        Returns True if request was found and removed, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired pending requests. Returns count removed."""
        pass


class SQLiteEnrollmentStore(EnrollmentStore):
    """SQLite-based enrollment store. Default for single-node deployments."""

    def __init__(self, db_path: str = "/var/lib/cert_service/enrollment.db"):
        """Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with self._connect() as conn:
            conn.executescript(
                """
                -- Enrolled entities (sites and users)
                CREATE TABLE IF NOT EXISTS enrolled_entities (
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL,
                    org TEXT,
                    role TEXT,
                    PRIMARY KEY (name, entity_type)
                );

                -- Pending enrollment requests
                CREATE TABLE IF NOT EXISTS pending_requests (
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    org TEXT NOT NULL,
                    csr_pem TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    token_subject TEXT NOT NULL,
                    role TEXT,
                    source_ip TEXT,
                    signed_cert TEXT,
                    approved INTEGER DEFAULT 0,
                    approved_at TEXT,
                    approved_by TEXT,
                    PRIMARY KEY (name, entity_type)
                );

                CREATE INDEX IF NOT EXISTS idx_pending_type
                    ON pending_requests(entity_type);
                CREATE INDEX IF NOT EXISTS idx_expires
                    ON pending_requests(expires_at);
            """
            )

    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────────────────
    # Enrolled Entities
    # ─────────────────────────────────────────────────────

    def is_enrolled(self, name: str, entity_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM enrolled_entities WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            ).fetchone()
        return row is not None

    def add_enrolled(self, entity: EnrolledEntity) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO enrolled_entities
                (name, entity_type, enrolled_at, org, role)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    entity.name,
                    entity.entity_type,
                    entity.enrolled_at.isoformat(),
                    entity.org,
                    entity.role,
                ),
            )
            # Remove from pending
            conn.execute(
                "DELETE FROM pending_requests WHERE name = ? AND entity_type = ?",
                (entity.name, entity.entity_type),
            )

    def get_enrolled(self, entity_type: Optional[str] = None) -> List[EnrolledEntity]:
        with self._connect() as conn:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM enrolled_entities WHERE entity_type = ?",
                    (entity_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM enrolled_entities").fetchall()
        return [self._row_to_entity(row) for row in rows]

    # ─────────────────────────────────────────────────────
    # Pending Requests
    # ─────────────────────────────────────────────────────

    def is_pending(self, name: str, entity_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM pending_requests WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            ).fetchone()
        return row is not None

    def add_pending(self, request: PendingRequest) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_requests
                (name, entity_type, org, csr_pem, submitted_at,
                 expires_at, token_subject, role, source_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    request.name,
                    request.entity_type,
                    request.org,
                    request.csr_pem,
                    request.submitted_at.isoformat(),
                    request.expires_at.isoformat(),
                    request.token_subject,
                    request.role,
                    request.source_ip,
                ),
            )

    def get_pending(self, name: str, entity_type: str) -> Optional[PendingRequest]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_requests WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            ).fetchone()
        if not row:
            return None
        return self._row_to_request(row)

    def get_all_pending(self, entity_type: Optional[str] = None) -> List[PendingRequest]:
        with self._connect() as conn:
            if entity_type:
                rows = conn.execute(
                    "SELECT * FROM pending_requests WHERE approved = 0 AND entity_type = ?",
                    (entity_type,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM pending_requests WHERE approved = 0").fetchall()
        return [self._row_to_request(row) for row in rows]

    def approve_pending(
        self,
        name: str,
        entity_type: str,
        signed_cert: str,
        approved_by: str,
    ) -> Optional[PendingRequest]:
        with self._connect() as conn:
            # Get the request first
            row = conn.execute(
                "SELECT * FROM pending_requests WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            ).fetchone()
            if not row:
                return None

            # Update as approved
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                UPDATE pending_requests
                SET signed_cert = ?, approved = 1, approved_at = ?, approved_by = ?
                WHERE name = ? AND entity_type = ?
            """,
                (signed_cert, now, approved_by, name, entity_type),
            )
            conn.commit()

            # Return updated request
            return self._row_to_request(row)

    def reject_pending(self, name: str, entity_type: str, reason: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_requests WHERE name = ? AND entity_type = ?",
                (name, entity_type),
            )
        return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_requests WHERE expires_at < ?",
                (now,),
            )
        return cursor.rowcount

    # ─────────────────────────────────────────────────────
    # Row Converters
    # ─────────────────────────────────────────────────────

    def _row_to_entity(self, row) -> EnrolledEntity:
        return EnrolledEntity(
            name=row["name"],
            entity_type=row["entity_type"],
            enrolled_at=datetime.fromisoformat(row["enrolled_at"]),
            org=row["org"],
            role=row["role"],
        )

    def _row_to_request(self, row) -> PendingRequest:
        return PendingRequest(
            name=row["name"],
            entity_type=row["entity_type"],
            org=row["org"],
            csr_pem=row["csr_pem"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            token_subject=row["token_subject"],
            role=row["role"],
            source_ip=row["source_ip"],
            signed_cert=row["signed_cert"],
            approved=bool(row["approved"]),
            approved_at=(datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None),
            approved_by=row["approved_by"],
        )


def create_enrollment_store(config: dict) -> EnrollmentStore:
    """Create enrollment store based on configuration.

    Args:
        config: Storage configuration dict with 'type' and type-specific options
                - type: 'sqlite' (default)
                - path: Database file path (for sqlite)

    Returns:
        EnrollmentStore instance
    """
    storage_type = config.get("type", "sqlite")

    if storage_type == "sqlite":
        path = config.get("path", "/var/lib/cert_service/enrollment.db")
        return SQLiteEnrollmentStore(db_path=path)

    elif storage_type == "postgresql":
        # Import only when needed (optional dependency)
        try:
            from nvflare.app_opt.cert_service.postgres_store import PostgreSQLEnrollmentStore

            conn_string = config["connection"]
            return PostgreSQLEnrollmentStore(connection_string=conn_string)
        except ImportError:
            raise ImportError("PostgreSQL support requires psycopg2. Install with: pip install psycopg2-binary")

    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


def approve_by_pattern(
    store: EnrollmentStore,
    pattern: str,
    entity_type: str,
    signed_cert_func,
    approved_by: str,
) -> List[str]:
    """Approve all pending requests matching a pattern.

    Args:
        store: EnrollmentStore instance
        pattern: Wildcard pattern (e.g., "hospital-*")
        entity_type: Entity type to match
        signed_cert_func: Function to generate signed cert for each request
        approved_by: Admin identifier

    Returns:
        List of approved entity names
    """
    pending = store.get_all_pending(entity_type)
    approved_names = []

    for req in pending:
        if fnmatch.fnmatch(req.name, pattern):
            signed_cert = signed_cert_func(req)
            store.approve_pending(req.name, entity_type, signed_cert, approved_by)
            approved_names.append(req.name)

    return approved_names
