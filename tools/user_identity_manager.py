# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import json
import os
import subprocess
from datetime import datetime, timezone

from helpers.tool import Tool, Response
from helpers import files

EI_IDENTITY_REPO_PATH = os.environ.get(
    "EI_IDENTITY_REPO_PATH",
    "/var/lib/agent-zero/ei-identity"
)

# Fallback for development
if not os.path.isdir(EI_IDENTITY_REPO_PATH):
    _alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ei-identity")
    if os.path.isdir(_alt):
        EI_IDENTITY_REPO_PATH = _alt


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dir(user_id: str) -> str:
    return os.path.join(EI_IDENTITY_REPO_PATH, "users", user_id)


def _account_path(user_id: str) -> str:
    return os.path.join(_user_dir(user_id), "account.json")


def _auth_config_path() -> str:
    return os.path.join(EI_IDENTITY_REPO_PATH, "config", "auth.json")


def _git_commit_and_push(message: str):
    """Stage all changes, commit, and push."""
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=EI_IDENTITY_REPO_PATH,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=EI_IDENTITY_REPO_PATH,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "push"],
            cwd=EI_IDENTITY_REPO_PATH,
            capture_output=True,
            timeout=60,
        )
    except Exception:
        pass  # Git push failures are non-fatal for the tool


class UserIdentityManager(Tool):
    """
    Manage user accounts in the ei-identity git repo.
    Handles registration, lookup, updates, and auth config.
    """

    async def execute(self, action="", user_id="", **kwargs):
        if not action:
            return Response(
                message="Error: 'action' required. Valid: register, get, update, list, deactivate, get_auth_config",
                break_loop=False,
            )

        action = action.strip().lower()

        if not os.path.isdir(EI_IDENTITY_REPO_PATH):
            return Response(
                message=f"Error: ei-identity repo not found at {EI_IDENTITY_REPO_PATH}. "
                        f"Clone smsmatt/ei-identity to this path first.",
                break_loop=False,
            )

        try:
            if action == "register":
                return await self._register(user_id, **kwargs)
            elif action == "get":
                return await self._get(user_id)
            elif action == "update":
                return await self._update(user_id, **kwargs)
            elif action == "list":
                return await self._list()
            elif action == "deactivate":
                return await self._deactivate(user_id)
            elif action == "get_auth_config":
                return await self._get_auth_config()
            else:
                return Response(
                    message=f"Unknown action '{action}'. Valid: register, get, update, list, deactivate, get_auth_config",
                    break_loop=False,
                )
        except Exception as e:
            return Response(message=f"user_identity_manager error: {e}", break_loop=False)

    async def _register(self, user_id: str, **kwargs) -> Response:
        """Register a new user."""
        if not user_id:
            return Response(message="Error: 'user_id' is required.", break_loop=False)

        account_path = _account_path(user_id)
        if os.path.exists(account_path):
            return Response(
                message=f"User '{user_id}' already exists. Use action='update' to modify.",
                break_loop=False,
            )

        display_name = kwargs.get("display_name", user_id)
        now = _now_iso()

        account = {
            "user_id": user_id,
            "display_name": display_name,
            "created_at": now,
            "active": True,
        }

        user_dir = _user_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)

        with open(account_path, "w") as f:
            json.dump(account, f, indent=2)

        # Create backup directory
        backup_dir = os.path.join(EI_IDENTITY_REPO_PATH, "backups", user_id)
        os.makedirs(backup_dir, exist_ok=True)
        gitkeep = os.path.join(backup_dir, ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, "w") as f:
                pass

        _git_commit_and_push(f"Register user: {user_id}")

        # Set in agent context
        self.agent.set_data("ei_user_id", user_id)

        return Response(
            message=f"Registered user '{user_id}' (display: {display_name}).",
            break_loop=False,
        )

    async def _get(self, user_id: str) -> Response:
        """Get user account info."""
        if not user_id:
            return Response(message="Error: 'user_id' is required.", break_loop=False)

        account_path = _account_path(user_id)
        if not os.path.exists(account_path):
            return Response(message=f"User '{user_id}' not found.", break_loop=False)

        with open(account_path, "r") as f:
            account = json.load(f)

        return Response(
            message=json.dumps(account, indent=2),
            break_loop=False,
        )

    async def _update(self, user_id: str, **kwargs) -> Response:
        """Update user account fields."""
        if not user_id:
            return Response(message="Error: 'user_id' is required.", break_loop=False)

        account_path = _account_path(user_id)
        if not os.path.exists(account_path):
            return Response(message=f"User '{user_id}' not found.", break_loop=False)

        with open(account_path, "r") as f:
            account = json.load(f)

        for field in ["display_name", "active"]:
            if field in kwargs and kwargs[field] != "":
                val = kwargs[field]
                if field == "active":
                    val = str(val).lower() in ("true", "yes", "1")
                account[field] = val

        with open(account_path, "w") as f:
            json.dump(account, f, indent=2)

        _git_commit_and_push(f"Update user: {user_id}")

        return Response(
            message=f"Updated user '{user_id}': {json.dumps(account, indent=2)}",
            break_loop=False,
        )

    async def _list(self) -> Response:
        """List all registered users."""
        users_dir = os.path.join(EI_IDENTITY_REPO_PATH, "users")
        if not os.path.isdir(users_dir):
            return Response(message="No users registered.", break_loop=False)

        users = []
        for name in sorted(os.listdir(users_dir)):
            account_path = os.path.join(users_dir, name, "account.json")
            if os.path.exists(account_path):
                with open(account_path, "r") as f:
                    account = json.load(f)
                status = "active" if account.get("active", False) else "inactive"
                users.append(f"- {account.get('user_id')} ({account.get('display_name')}) [{status}]")

        if not users:
            return Response(message="No users registered.", break_loop=False)

        return Response(
            message=f"Registered users ({len(users)}):\n" + "\n".join(users),
            break_loop=False,
        )

    async def _deactivate(self, user_id: str) -> Response:
        """Deactivate a user (soft delete)."""
        if not user_id:
            return Response(message="Error: 'user_id' is required.", break_loop=False)

        account_path = _account_path(user_id)
        if not os.path.exists(account_path):
            return Response(message=f"User '{user_id}' not found.", break_loop=False)

        with open(account_path, "r") as f:
            account = json.load(f)

        account["active"] = False

        with open(account_path, "w") as f:
            json.dump(account, f, indent=2)

        _git_commit_and_push(f"Deactivate user: {user_id}")

        return Response(
            message=f"User '{user_id}' deactivated.",
            break_loop=False,
        )

    async def _get_auth_config(self) -> Response:
        """Get current auth configuration."""
        config_path = _auth_config_path()
        if not os.path.exists(config_path):
            return Response(message="Auth config not found.", break_loop=False)

        with open(config_path, "r") as f:
            config = json.load(f)

        return Response(
            message=f"Auth configuration:\n{json.dumps(config, indent=2)}",
            break_loop=False,
        )
