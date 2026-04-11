# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

from python.helpers.extension import Extension
from python.helpers.memory import Memory
from agent import LoopData
from python.helpers import errors


class EIEmotionalContext(Extension):
    """
    Inject current emotional context from FAISS FRAGMENTS into the message loop.
    Runs after memory recall to provide the LLM with grounded emotional state.
    """

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # Only run on first iteration and every 3rd iteration after
        if loop_data.iteration > 1 and loop_data.iteration % 3 != 0:
            return

        try:
            db = await Memory.get(self.agent)

            # Search for recent Webb Equations in this session
            results = await db.search_similarity_threshold(
                query="Webb Equation emotion analysis emotional reaction",
                limit=5,
                threshold=0.3,
                filter=f"area == '{Memory.Area.FRAGMENTS.value}' and ei_type == 'webb_equation'",
            )

            if not results:
                return

            # Build compact emotional context
            lines = ["## Active Emotional Context (Webb Equations this session)"]
            for doc in results:
                lines.append(f"- {doc.page_content}")

            # Also include {self} map summary if loaded
            user_id = self.agent.get_data("ei_user_id")
            if user_id:
                self_map = self.agent.get_data(f"ei_self_map_{user_id}")
                if self_map:
                    high_power = [
                        item for item in self_map
                        if int(item.get("power_level", 0)) >= 7
                    ]
                    if high_power:
                        lines.append("\n**Highest-power {self} attachments:**")
                        for item in sorted(high_power, key=lambda x: int(x.get("power_level", 0)), reverse=True)[:5]:
                            lines.append(
                                f"- {item.get('label')} ({item.get('quadrant')}, power={item.get('power_level')})"
                            )

            context_text = "\n".join(lines)
            loop_data.extras_persistent["ei_emotional_context"] = context_text

        except Exception as e:
            # Non-fatal: emotional context is supplementary
            self.agent.context.log.log(
                type="warning",
                heading="EI emotional context error",
                content=errors.format_error(e),
            )
