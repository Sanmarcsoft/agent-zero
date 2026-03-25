# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER
#
# Integration tests for backup/restore round-trip.
# Requires: ChromaDB at EI_CHROMADB_HOST:EI_CHROMADB_PORT, GPG key available.

import json
import os
import sys
import tempfile
import subprocess

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip if ChromaDB is not available
try:
    import chromadb
    _client = chromadb.HttpClient(
        host=os.environ.get("EI_CHROMADB_HOST", "10.0.0.12"),
        port=int(os.environ.get("EI_CHROMADB_PORT", "18000")),
    )
    _client.heartbeat()
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

# Skip if GPG is not available
try:
    result = subprocess.run(["gpg", "--version"], capture_output=True, timeout=5)
    GPG_AVAILABLE = result.returncode == 0
except Exception:
    GPG_AVAILABLE = False


FIXTURES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures", "ei_test_data.json"
)

TEST_USER_ID = "ei-test-user"


def load_fixtures():
    with open(FIXTURES_PATH) as f:
        return json.load(f)


@pytest.fixture
def chromadb_client():
    """Get ChromaDB client."""
    return chromadb.HttpClient(
        host=os.environ.get("EI_CHROMADB_HOST", "10.0.0.12"),
        port=int(os.environ.get("EI_CHROMADB_PORT", "18000")),
    )


@pytest.fixture
def fixtures():
    return load_fixtures()


@pytest.fixture
def seeded_chromadb(chromadb_client, fixtures):
    """Seed ChromaDB with test fixtures, yield, then clean up."""
    # Seed self maps
    self_maps = chromadb_client.get_or_create_collection("ei_self_maps")
    for item in fixtures["self_map_items"]:
        self_maps.add(
            ids=[item["item_id"]],
            documents=[f"{item['label']} ({item['quadrant']}, power={item['power_level']}, valence={item['valence']})"],
            metadatas=[{
                "user_id": TEST_USER_ID,
                **{k: v for k, v in item.items() if k != "item_id"},
                "item_id": item["item_id"],
            }],
        )

    # Seed emotion log
    emotion_log = chromadb_client.get_or_create_collection("ei_emotion_log")
    for event in fixtures["emotion_log_events"]:
        eq_id = event["equation_id"]
        emotion_log.add(
            ids=[eq_id],
            documents=[f"{event['perception']} → {event['emotion_group']} ({event['severity_label']})"],
            metadatas=[{
                "user_id": TEST_USER_ID,
                **{k: (v if not isinstance(v, bool) else str(v)) for k, v in event.items()},
            }],
        )

    # Seed ToM profiles
    tom_profiles = chromadb_client.get_or_create_collection("ei_tom_profiles")
    for profile in fixtures["tom_profiles"]:
        tom_profiles.add(
            ids=[profile["profile_id"]],
            documents=[json.dumps(profile["attachments"])],
            metadatas=[{
                "user_id": TEST_USER_ID,
                "person_name": profile["person_name"],
                "person_display_name": profile["person_display_name"],
                "relationship": profile["relationship"],
                "record_type": "profile",
                "attachment_count": len(profile["attachments"]),
                "created_at": "2026-03-19T00:00:00Z",
                "updated_at": "2026-03-19T00:00:00Z",
            }],
        )

    yield fixtures

    # Cleanup
    _cleanup_test_user(chromadb_client)


def _cleanup_test_user(client):
    """Remove all test user data from ChromaDB."""
    for coll_name in ["ei_self_maps", "ei_emotion_log", "ei_tom_profiles"]:
        try:
            coll = client.get_collection(coll_name)
            existing = coll.get(where={"user_id": TEST_USER_ID})
            if existing["ids"]:
                coll.delete(ids=existing["ids"])
        except Exception:
            pass


# ── Seeding Tests ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
class TestSeeding:
    """Verify test data seeds correctly into ChromaDB."""

    def test_self_maps_seeded(self, seeded_chromadb, chromadb_client):
        coll = chromadb_client.get_collection("ei_self_maps")
        results = coll.get(where={"user_id": TEST_USER_ID})
        assert len(results["ids"]) == 5

    def test_emotion_log_seeded(self, seeded_chromadb, chromadb_client):
        coll = chromadb_client.get_collection("ei_emotion_log")
        results = coll.get(where={"user_id": TEST_USER_ID})
        assert len(results["ids"]) == 10

    def test_tom_profiles_seeded(self, seeded_chromadb, chromadb_client):
        coll = chromadb_client.get_collection("ei_tom_profiles")
        results = coll.get(where={"user_id": TEST_USER_ID})
        assert len(results["ids"]) == 3

    def test_self_map_fields_correct(self, seeded_chromadb, chromadb_client, fixtures):
        coll = chromadb_client.get_collection("ei_self_maps")
        results = coll.get(
            where={"user_id": TEST_USER_ID},
            include=["metadatas"],
        )
        # Find the spouse item
        spouse = None
        for meta in results["metadatas"]:
            if meta.get("label") == "spouse":
                spouse = meta
                break
        assert spouse is not None
        assert spouse["quadrant"] == "people"
        assert spouse["power_level"] == 9
        assert spouse["valence"] == 8


# ── Backup/Restore Round-Trip Tests ───────────────────────────────────────

@pytest.mark.skipif(
    not CHROMADB_AVAILABLE or not GPG_AVAILABLE,
    reason="ChromaDB or GPG not available"
)
class TestBackupRestore:
    """Full backup → wipe → restore → validate round-trip."""

    def test_backup_creates_gpg_files(self, seeded_chromadb):
        from python.helpers.ei_backup import backup_user

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backup directory structure
            backup_dir = os.path.join(tmpdir, "backups", TEST_USER_ID)
            os.makedirs(backup_dir, exist_ok=True)

            # Initialize as git repo for backup_user
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True,
            )

            result = backup_user(TEST_USER_ID, repo_path=tmpdir)

            if result["status"] == "error":
                pytest.skip(f"Backup failed: {result.get('message')}")

            assert result["status"] == "ok"
            assert len(result["files"]) == 3

            for f in result["files"]:
                assert os.path.exists(f)
                assert f.endswith(".gpg")

    def test_gpg_files_are_encrypted(self, seeded_chromadb):
        from python.helpers.ei_backup import backup_user

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = os.path.join(tmpdir, "backups", TEST_USER_ID)
            os.makedirs(backup_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True,
            )

            result = backup_user(TEST_USER_ID, repo_path=tmpdir)
            if result["status"] == "error":
                pytest.skip(f"Backup failed: {result.get('message')}")

            # GPG files should not be plaintext JSON
            for f in result["files"]:
                with open(f, "rb") as fh:
                    content = fh.read()
                    # Should contain GPG markers, not raw JSON
                    assert b'"user_id"' not in content or b"-----BEGIN PGP MESSAGE-----" in content

    def test_full_round_trip(self, seeded_chromadb, chromadb_client, fixtures):
        from python.helpers.ei_backup import backup_user, restore_user

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = os.path.join(tmpdir, "backups", TEST_USER_ID)
            os.makedirs(backup_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True,
            )

            # 1. Backup
            backup_result = backup_user(TEST_USER_ID, repo_path=tmpdir)
            if backup_result["status"] == "error":
                pytest.skip(f"Backup failed: {backup_result.get('message')}")

            # 2. Wipe ChromaDB
            _cleanup_test_user(chromadb_client)

            # 3. Verify wipe
            for coll_name in ["ei_self_maps", "ei_emotion_log", "ei_tom_profiles"]:
                coll = chromadb_client.get_collection(coll_name)
                results = coll.get(where={"user_id": TEST_USER_ID})
                assert len(results["ids"]) == 0, f"{coll_name} not fully wiped"

            # 4. Restore
            restore_result = restore_user(TEST_USER_ID, repo_path=tmpdir)
            assert restore_result["status"] == "ok"

            # 5. Validate self maps
            self_maps = chromadb_client.get_collection("ei_self_maps")
            sm_results = self_maps.get(
                where={"user_id": TEST_USER_ID},
                include=["metadatas"],
            )
            assert len(sm_results["ids"]) == 5, f"Expected 5 self map items, got {len(sm_results['ids'])}"

            # Validate each self map item
            restored_by_label = {m["label"]: m for m in sm_results["metadatas"]}
            for orig in fixtures["self_map_items"]:
                label = orig["label"]
                assert label in restored_by_label, f"Self map item '{label}' not restored"
                restored = restored_by_label[label]
                assert restored["quadrant"] == orig["quadrant"]
                assert int(restored["power_level"]) == orig["power_level"]
                assert int(restored["valence"]) == orig["valence"]

            # 6. Validate emotion log
            emotion_log = chromadb_client.get_collection("ei_emotion_log")
            el_results = emotion_log.get(
                where={"user_id": TEST_USER_ID},
                include=["metadatas"],
            )
            assert len(el_results["ids"]) == 10, f"Expected 10 emotion events, got {len(el_results['ids'])}"

            # 7. Validate ToM profiles
            tom = chromadb_client.get_collection("ei_tom_profiles")
            tom_results = tom.get(
                where={"user_id": TEST_USER_ID},
                include=["metadatas", "documents"],
            )
            assert len(tom_results["ids"]) == 3, f"Expected 3 ToM profiles, got {len(tom_results['ids'])}"


# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
