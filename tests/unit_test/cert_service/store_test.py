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

"""Unit tests for enrollment store (store.py)."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from nvflare.cert_service.store import EnrolledEntity, PendingRequest, SQLiteEnrollmentStore, create_enrollment_store


class TestSQLiteEnrollmentStore:
    """Tests for SQLiteEnrollmentStore."""

    @pytest.fixture
    def store(self):
        """Create a temporary SQLite store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield SQLiteEnrollmentStore(db_path)

    @pytest.fixture
    def sample_entity(self):
        """Create a sample enrolled entity."""
        return EnrolledEntity(
            name="site-1",
            entity_type="client",
            enrolled_at=datetime.now(timezone.utc),
            org="TestOrg",
            role=None,
        )

    @pytest.fixture
    def sample_pending(self):
        """Create a sample pending request."""
        now = datetime.now(timezone.utc)
        return PendingRequest(
            name="site-2",
            entity_type="client",
            org="TestOrg",
            csr_pem="-----BEGIN CERTIFICATE REQUEST-----\ntest\n-----END CERTIFICATE REQUEST-----",
            submitted_at=now,
            expires_at=now + timedelta(days=7),
            token_subject="site-2",
            role=None,
            source_ip="10.0.0.1",
        )


class TestEnrolledEntities(TestSQLiteEnrollmentStore):
    """Tests for enrolled entity operations."""

    def test_add_and_check_enrolled(self, store, sample_entity):
        """Test adding and checking enrolled entity."""
        assert not store.is_enrolled("site-1", "client")
        store.add_enrolled(sample_entity)
        assert store.is_enrolled("site-1", "client")

    def test_is_enrolled_different_type(self, store, sample_entity):
        """Test that entity type matters for enrollment check."""
        store.add_enrolled(sample_entity)
        assert store.is_enrolled("site-1", "client")
        assert not store.is_enrolled("site-1", "relay")
        assert not store.is_enrolled("site-1", "admin")

    def test_get_enrolled_all(self, store):
        """Test getting all enrolled entities."""
        entities = [
            EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "OrgA"),
            EnrolledEntity("site-2", "client", datetime.now(timezone.utc), "OrgB"),
            EnrolledEntity("relay-1", "relay", datetime.now(timezone.utc), "OrgA"),
            EnrolledEntity("admin-1", "admin", datetime.now(timezone.utc), "OrgA", "lead"),
        ]
        for entity in entities:
            store.add_enrolled(entity)

        result = store.get_enrolled()
        assert len(result) == 4

    def test_get_enrolled_filtered_by_type(self, store):
        """Test getting enrolled entities filtered by type."""
        entities = [
            EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "OrgA"),
            EnrolledEntity("site-2", "client", datetime.now(timezone.utc), "OrgB"),
            EnrolledEntity("relay-1", "relay", datetime.now(timezone.utc), "OrgA"),
        ]
        for entity in entities:
            store.add_enrolled(entity)

        clients = store.get_enrolled("client")
        assert len(clients) == 2

        relays = store.get_enrolled("relay")
        assert len(relays) == 1

        admins = store.get_enrolled("admin")
        assert len(admins) == 0

    def test_enrolled_entity_with_role(self, store):
        """Test enrolled entity with admin role."""
        entity = EnrolledEntity(
            name="admin-1",
            entity_type="admin",
            enrolled_at=datetime.now(timezone.utc),
            org="TestOrg",
            role="lead",
        )
        store.add_enrolled(entity)

        result = store.get_enrolled("admin")
        assert len(result) == 1
        assert result[0].role == "lead"

    def test_add_enrolled_removes_pending(self, store, sample_pending):
        """Test that adding enrolled entity removes it from pending."""
        # Add as pending first
        store.add_pending(sample_pending)
        assert store.is_pending("site-2", "client")

        # Enroll
        entity = EnrolledEntity(
            name="site-2",
            entity_type="client",
            enrolled_at=datetime.now(timezone.utc),
            org="TestOrg",
        )
        store.add_enrolled(entity)

        # Should be enrolled, not pending
        assert store.is_enrolled("site-2", "client")
        assert not store.is_pending("site-2", "client")

    def test_duplicate_enrollment_replaces(self, store):
        """Test that re-enrollment replaces existing record."""
        entity1 = EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "OrgA")
        store.add_enrolled(entity1)

        entity2 = EnrolledEntity("site-1", "client", datetime.now(timezone.utc), "OrgB")
        store.add_enrolled(entity2)

        result = store.get_enrolled("client")
        assert len(result) == 1
        assert result[0].org == "OrgB"


class TestPendingRequests(TestSQLiteEnrollmentStore):
    """Tests for pending request operations."""

    def test_add_and_check_pending(self, store, sample_pending):
        """Test adding and checking pending request."""
        assert not store.is_pending("site-2", "client")
        store.add_pending(sample_pending)
        assert store.is_pending("site-2", "client")

    def test_is_pending_different_type(self, store, sample_pending):
        """Test that entity type matters for pending check."""
        store.add_pending(sample_pending)
        assert store.is_pending("site-2", "client")
        assert not store.is_pending("site-2", "relay")

    def test_get_pending(self, store, sample_pending):
        """Test getting a specific pending request."""
        store.add_pending(sample_pending)

        result = store.get_pending("site-2", "client")
        assert result is not None
        assert result.name == "site-2"
        assert result.entity_type == "client"
        assert result.org == "TestOrg"
        assert result.source_ip == "10.0.0.1"

    def test_get_pending_not_found(self, store):
        """Test getting non-existent pending request."""
        result = store.get_pending("nonexistent", "client")
        assert result is None

    def test_get_all_pending(self, store):
        """Test getting all pending requests."""
        now = datetime.now(timezone.utc)
        requests = [
            PendingRequest("site-1", "client", "Org", "csr1", now, now + timedelta(days=7), "site-1"),
            PendingRequest("site-2", "client", "Org", "csr2", now, now + timedelta(days=7), "site-2"),
            PendingRequest("relay-1", "relay", "Org", "csr3", now, now + timedelta(days=7), "relay-1"),
        ]
        for req in requests:
            store.add_pending(req)

        all_pending = store.get_all_pending()
        assert len(all_pending) == 3

    def test_get_all_pending_filtered_by_type(self, store):
        """Test getting pending requests filtered by type."""
        now = datetime.now(timezone.utc)
        requests = [
            PendingRequest("site-1", "client", "Org", "csr1", now, now + timedelta(days=7), "site-1"),
            PendingRequest("site-2", "client", "Org", "csr2", now, now + timedelta(days=7), "site-2"),
            PendingRequest("relay-1", "relay", "Org", "csr3", now, now + timedelta(days=7), "relay-1"),
        ]
        for req in requests:
            store.add_pending(req)

        clients = store.get_all_pending("client")
        assert len(clients) == 2

        relays = store.get_all_pending("relay")
        assert len(relays) == 1

    def test_approve_pending(self, store, sample_pending):
        """Test approving a pending request."""
        store.add_pending(sample_pending)

        result = store.approve_pending(
            name="site-2",
            entity_type="client",
            signed_cert="-----BEGIN CERTIFICATE-----\nsigned\n-----END CERTIFICATE-----",
            approved_by="admin",
        )

        assert result is not None
        assert result.name == "site-2"

        # After approval, should still be retrievable with signed cert
        pending = store.get_pending("site-2", "client")
        assert pending.approved is True
        assert pending.signed_cert is not None

    def test_approve_pending_not_found(self, store):
        """Test approving non-existent pending request."""
        result = store.approve_pending("nonexistent", "client", "cert", "admin")
        assert result is None

    def test_reject_pending(self, store, sample_pending):
        """Test rejecting a pending request."""
        store.add_pending(sample_pending)
        assert store.is_pending("site-2", "client")

        removed = store.reject_pending("site-2", "client", "Not authorized")
        assert removed is True
        assert not store.is_pending("site-2", "client")

    def test_reject_pending_not_found(self, store):
        """Test rejecting non-existent pending request."""
        removed = store.reject_pending("nonexistent", "client", "reason")
        assert removed is False

    def test_approved_not_in_all_pending(self, store, sample_pending):
        """Test that approved requests are excluded from get_all_pending."""
        store.add_pending(sample_pending)
        assert len(store.get_all_pending()) == 1

        store.approve_pending("site-2", "client", "cert", "admin")
        assert len(store.get_all_pending()) == 0


class TestCleanupExpired(TestSQLiteEnrollmentStore):
    """Tests for cleanup_expired operation."""

    def test_cleanup_expired(self, store):
        """Test cleaning up expired pending requests."""
        now = datetime.now(timezone.utc)

        # Add expired request
        expired = PendingRequest(
            "site-old", "client", "Org", "csr", now - timedelta(days=10), now - timedelta(days=3), "site-old"  # Expired
        )
        store.add_pending(expired)

        # Add valid request
        valid = PendingRequest("site-new", "client", "Org", "csr", now, now + timedelta(days=7), "site-new")
        store.add_pending(valid)

        # Cleanup should remove expired
        removed = store.cleanup_expired()
        assert removed == 1

        # Valid should remain
        assert store.is_pending("site-new", "client")
        assert not store.is_pending("site-old", "client")


class TestCreateEnrollmentStore:
    """Tests for create_enrollment_store factory function."""

    def test_create_sqlite_store(self):
        """Test creating SQLite store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "type": "sqlite",
                "path": os.path.join(tmpdir, "test.db"),
            }
            store = create_enrollment_store(config)
            assert isinstance(store, SQLiteEnrollmentStore)

    def test_create_store_default_type(self):
        """Test creating store with default type (sqlite)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "path": os.path.join(tmpdir, "test.db"),
            }
            store = create_enrollment_store(config)
            assert isinstance(store, SQLiteEnrollmentStore)

    def test_create_store_unknown_type(self):
        """Test creating store with unknown type raises error."""
        config = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown storage type"):
            create_enrollment_store(config)


class TestEntityUniqueness(TestSQLiteEnrollmentStore):
    """Tests for entity uniqueness by (name, entity_type)."""

    def test_same_name_different_types_enrolled(self, store):
        """Test same name can be enrolled as different types."""
        client = EnrolledEntity("entity-1", "client", datetime.now(timezone.utc), "Org")
        admin = EnrolledEntity("entity-1", "admin", datetime.now(timezone.utc), "Org", "lead")

        store.add_enrolled(client)
        store.add_enrolled(admin)

        assert store.is_enrolled("entity-1", "client")
        assert store.is_enrolled("entity-1", "admin")

        all_enrolled = store.get_enrolled()
        assert len(all_enrolled) == 2

    def test_same_name_different_types_pending(self, store):
        """Test same name can have pending requests as different types."""
        now = datetime.now(timezone.utc)

        client = PendingRequest("entity-1", "client", "Org", "csr1", now, now + timedelta(days=7), "entity-1")
        relay = PendingRequest("entity-1", "relay", "Org", "csr2", now, now + timedelta(days=7), "entity-1")

        store.add_pending(client)
        store.add_pending(relay)

        assert store.is_pending("entity-1", "client")
        assert store.is_pending("entity-1", "relay")

        all_pending = store.get_all_pending()
        assert len(all_pending) == 2
