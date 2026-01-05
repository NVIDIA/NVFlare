# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""PostgreSQL-based Enrollment Store for multi-instance deployments.

This is an OPTIONAL plugin for high-availability Certificate Service deployments.
It requires the psycopg2 library:

    pip install psycopg2-binary

Usage:
    # In cert_service_config.yaml
    storage:
      type: postgresql
      connection: "postgresql://user:pass@host:5432/certservice"
      pool_size: 10

Features:
- Connection pooling for high concurrency
- Multi-instance support (multiple Certificate Service instances can share DB)
- Row-level locking for safe concurrent writes
"""

from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

try:
    import psycopg2
    from psycopg2 import pool
except ImportError:
    raise ImportError("PostgreSQL support requires psycopg2. " "Install with: pip install psycopg2-binary")

from nvflare.cert_service.store import EnrolledEntity, EnrollmentStore, PendingRequest


class PostgreSQLEnrollmentStore(EnrollmentStore):
    """PostgreSQL-based enrollment store for multi-instance deployments.

    This store supports connection pooling and is designed for high-availability
    deployments where multiple Certificate Service instances share the same database.
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        """Initialize PostgreSQL store with connection pooling.

        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:pass@host:5432/certservice"
            pool_size: Minimum connections in pool (default: 10)
            max_overflow: Maximum additional connections beyond pool_size (default: 20)
        """
        self.connection_string = connection_string
        self._pool = pool.ThreadedConnectionPool(
            minconn=pool_size,
            maxconn=pool_size + max_overflow,
            dsn=connection_string,
        )
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    -- Enrolled entities (sites and users)
                    CREATE TABLE IF NOT EXISTS enrolled_entities (
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        enrolled_at TIMESTAMP NOT NULL,
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
                        submitted_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        token_subject TEXT NOT NULL,
                        role TEXT,
                        source_ip TEXT,
                        signed_cert TEXT,
                        approved BOOLEAN DEFAULT FALSE,
                        approved_at TIMESTAMP,
                        approved_by TEXT,
                        PRIMARY KEY (name, entity_type)
                    );

                    CREATE INDEX IF NOT EXISTS idx_pending_type
                        ON pending_requests(entity_type);
                    CREATE INDEX IF NOT EXISTS idx_expires
                        ON pending_requests(expires_at);
                    CREATE INDEX IF NOT EXISTS idx_enrolled_type
                        ON enrolled_entities(entity_type);
                    """
                )
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get connection from pool with automatic return."""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    # ─────────────────────────────────────────────────────
    # Enrolled Entities
    # ─────────────────────────────────────────────────────

    def is_enrolled(self, name: str, entity_type: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM enrolled_entities WHERE name = %s AND entity_type = %s",
                    (name, entity_type),
                )
                return cur.fetchone() is not None

    def add_enrolled(self, entity: EnrolledEntity) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # Upsert enrolled entity
                cur.execute(
                    """
                    INSERT INTO enrolled_entities (name, entity_type, enrolled_at, org, role)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name, entity_type) DO UPDATE SET
                        enrolled_at = EXCLUDED.enrolled_at,
                        org = EXCLUDED.org,
                        role = EXCLUDED.role
                    """,
                    (
                        entity.name,
                        entity.entity_type,
                        entity.enrolled_at,
                        entity.org,
                        entity.role,
                    ),
                )
                # Remove from pending
                cur.execute(
                    "DELETE FROM pending_requests WHERE name = %s AND entity_type = %s",
                    (entity.name, entity.entity_type),
                )
            conn.commit()

    def get_enrolled(self, entity_type: Optional[str] = None) -> List[EnrolledEntity]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if entity_type:
                    cur.execute(
                        "SELECT name, entity_type, enrolled_at, org, role FROM enrolled_entities WHERE entity_type = %s",
                        (entity_type,),
                    )
                else:
                    cur.execute("SELECT name, entity_type, enrolled_at, org, role FROM enrolled_entities")
                rows = cur.fetchall()
        return [
            EnrolledEntity(
                name=row[0],
                entity_type=row[1],
                enrolled_at=row[2],
                org=row[3],
                role=row[4],
            )
            for row in rows
        ]

    # ─────────────────────────────────────────────────────
    # Pending Requests
    # ─────────────────────────────────────────────────────

    def is_pending(self, name: str, entity_type: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pending_requests WHERE name = %s AND entity_type = %s",
                    (name, entity_type),
                )
                return cur.fetchone() is not None

    def add_pending(self, request: PendingRequest) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pending_requests
                    (name, entity_type, org, csr_pem, submitted_at, expires_at, token_subject, role, source_ip)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name, entity_type) DO UPDATE SET
                        org = EXCLUDED.org,
                        csr_pem = EXCLUDED.csr_pem,
                        submitted_at = EXCLUDED.submitted_at,
                        expires_at = EXCLUDED.expires_at,
                        token_subject = EXCLUDED.token_subject,
                        role = EXCLUDED.role,
                        source_ip = EXCLUDED.source_ip
                    """,
                    (
                        request.name,
                        request.entity_type,
                        request.org,
                        request.csr_pem,
                        request.submitted_at,
                        request.expires_at,
                        request.token_subject,
                        request.role,
                        request.source_ip,
                    ),
                )
            conn.commit()

    def get_pending(self, name: str, entity_type: str) -> Optional[PendingRequest]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT name, entity_type, org, csr_pem, submitted_at, expires_at,
                              token_subject, role, source_ip, signed_cert, approved, approved_at, approved_by
                       FROM pending_requests WHERE name = %s AND entity_type = %s""",
                    (name, entity_type),
                )
                row = cur.fetchone()
        if not row:
            return None
        return self._row_to_request(row)

    def get_all_pending(self, entity_type: Optional[str] = None) -> List[PendingRequest]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if entity_type:
                    cur.execute(
                        """SELECT name, entity_type, org, csr_pem, submitted_at, expires_at,
                                  token_subject, role, source_ip, signed_cert, approved, approved_at, approved_by
                           FROM pending_requests WHERE approved = FALSE AND entity_type = %s""",
                        (entity_type,),
                    )
                else:
                    cur.execute(
                        """SELECT name, entity_type, org, csr_pem, submitted_at, expires_at,
                                  token_subject, role, source_ip, signed_cert, approved, approved_at, approved_by
                           FROM pending_requests WHERE approved = FALSE"""
                    )
                rows = cur.fetchall()
        return [self._row_to_request(row) for row in rows]

    def approve_pending(
        self,
        name: str,
        entity_type: str,
        signed_cert: str,
        approved_by: str,
    ) -> Optional[PendingRequest]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # Get and lock the row
                cur.execute(
                    """SELECT name, entity_type, org, csr_pem, submitted_at, expires_at,
                              token_subject, role, source_ip, signed_cert, approved, approved_at, approved_by
                       FROM pending_requests
                       WHERE name = %s AND entity_type = %s
                       FOR UPDATE""",
                    (name, entity_type),
                )
                row = cur.fetchone()
                if not row:
                    return None

                # Update as approved
                now = datetime.utcnow()
                cur.execute(
                    """UPDATE pending_requests
                       SET signed_cert = %s, approved = TRUE, approved_at = %s, approved_by = %s
                       WHERE name = %s AND entity_type = %s""",
                    (signed_cert, now, approved_by, name, entity_type),
                )
            conn.commit()
            return self._row_to_request(row)

    def reject_pending(self, name: str, entity_type: str, reason: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM pending_requests WHERE name = %s AND entity_type = %s",
                    (name, entity_type),
                )
                count = cur.rowcount
            conn.commit()
        return count > 0

    def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM pending_requests WHERE expires_at < %s",
                    (now,),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def _row_to_request(self, row) -> PendingRequest:
        return PendingRequest(
            name=row[0],
            entity_type=row[1],
            org=row[2],
            csr_pem=row[3],
            submitted_at=row[4],
            expires_at=row[5],
            token_subject=row[6],
            role=row[7],
            source_ip=row[8],
            signed_cert=row[9],
            approved=row[10],
            approved_at=row[11],
            approved_by=row[12],
        )

    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
