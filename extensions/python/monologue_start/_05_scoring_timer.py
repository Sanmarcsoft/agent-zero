# Scoring Timer — records monologue start time for latency calculation
import time
from helpers.extension import Extension
from agent import LoopData


class ScoringTimer(Extension):

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        self.agent._monologue_start_time = time.time()
