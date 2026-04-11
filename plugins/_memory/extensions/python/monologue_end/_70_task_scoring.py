# Task Scoring Extension
# Records task performance metrics at the end of each monologue.
# Aligns with Zorin's TaskRecord schema for unified agent scoring.
#
# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import time
import re
from helpers.extension import Extension
from helpers.print_style import PrintStyle
from agent import LoopData


class TaskScoring(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        # Only score primary agent (agent 0) interactions
        if self.agent.number != 0:
            return

        # Calculate basic metrics from the monologue
        iterations = loop_data.iteration
        if iterations <= 0:
            return

        # Import the scoring API's history storage
        try:
            from python.api.agent_scoring import _task_history, _MAX_HISTORY
        except ImportError:
            return

        global _task_counter
        try:
            from python.api.agent_scoring import _task_counter as counter
        except ImportError:
            counter = len(_task_history)

        # Extract the user's original message
        user_msg = loop_data.user_message or ""
        if len(user_msg) > 200:
            user_msg = user_msg[:200] + "..."

        # Calculate latency from loop iterations (rough proxy)
        # The actual latency is tracked by the monologue start/end
        latency_ms = 0
        if hasattr(self.agent, '_monologue_start_time'):
            latency_ms = int((time.time() - self.agent._monologue_start_time) * 1000)

        # Extract any Webb equation from the agent's response history
        webb_equation = None
        for msg in reversed(self.agent.history):
            content = str(msg.get("content", ""))
            webb_match = re.search(r'\[WEBB\][^\n]*', content)
            if webb_match:
                webb_equation = webb_match.group(0)
                break

        # Self-scoring: based on whether the agent completed without errors
        # Format errors reduce score, successful tool execution increases it
        score = 7  # baseline for successful completion
        if iterations == 1:
            score = 8  # completed in one shot
        elif iterations <= 3:
            score = 7  # reasonable
        elif iterations <= 5:
            score = 6  # took some retries
        else:
            score = 5  # many retries, likely had format issues

        # Check for format errors in history (reduce score)
        format_errors = sum(
            1 for msg in self.agent.history
            if "misformatted" in str(msg.get("content", "")).lower()
        )
        score = max(1, score - format_errors)

        record = {
            "id": f"QT-{len(_task_history) + 1}-{int(time.time() * 1000)}",
            "task": user_msg,
            "response": f"Completed in {iterations} iteration(s)",
            "score": score,
            "latency_ms": latency_ms,
            "tokens": 0,  # TODO: track actual token usage from LLM response
            "tokens_per_second": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spoke": False,
            "webb_equation": webb_equation,
            "error": False,
        }

        _task_history.append(record)
        if len(_task_history) > _MAX_HISTORY:
            _task_history.pop(0)

        PrintStyle(font_color="#4a90d9", padding=False).print(
            f"[Scoring] Task scored {score}/10 ({iterations} iterations)"
        )
