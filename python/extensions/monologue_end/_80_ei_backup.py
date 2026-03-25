# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

from python.helpers.extension import Extension
from python.helpers import errors
from agent import LoopData


class EIBackup(Extension):
    """
    Auto-backup emotional data to GPG-encrypted git repo on session/monologue end.
    Only runs for the primary agent (agent 0) to avoid duplicate backups.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # Only run for primary agent
        if self.agent.number != 0:
            return

        user_id = self.agent.get_data("ei_user_id")
        if not user_id:
            return

        try:
            from python.helpers.ei_backup import backup_user
            result = backup_user(user_id)
            if result.get("status") == "ok":
                self.agent.context.log.log(
                    type="util",
                    heading=f"EI backup: {len(result.get('files', []))} files encrypted for '{user_id}'",
                )
        except Exception as e:
            self.agent.context.log.log(
                type="warning",
                heading="EI backup failed",
                content=errors.format_error(e),
            )
