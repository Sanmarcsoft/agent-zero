## Theory of Mind (MHH-EI)
model third-party emotional states using embedded {self} maps and perception tracking
use when user asks how someone else feels or in multi-party scenarios

### theory_of_mind
Actions: analyze | create_profile | update_profile | get_profile | perception_check
- person_name: who to model (required)
- situation: event or scenario (required for analyze)
- has_perception: does this person know about it? (true/false) — KEY for ToM
- affected_attachment: which of their attachments is affected
- ep_power, p_weight, valence_shift, source, time_frame: Webb Equation params from their perspective

Analyze someone's feelings:
~~~json
{
    "thoughts": [
        "User asks how their wife feels about the move. I need to model wife's perspective.",
        "Her {self} map likely includes their home, her friends, her job in the current city."
    ],
    "headline": "Modeling wife's emotional state",
    "tool_name": "theory_of_mind",
    "tool_args": {
        "action": "analyze",
        "person_name": "wife",
        "situation": "Potential relocation to a new city",
        "has_perception": "true",
        "affected_attachment": "home",
        "valence_shift": "negative",
        "source": "external",
        "time_frame": "future",
        "ep_power": "8",
        "p_weight": "7"
    }
}
~~~

Perception check (Sally-Anne test pattern):
~~~json
{
    "thoughts": [
        "Does this person actually know about the event? Perception is the linchpin of ToM."
    ],
    "headline": "Checking if person has perception of event",
    "tool_name": "theory_of_mind",
    "tool_args": {
        "action": "perception_check",
        "person_name": "Sally",
        "event": "Anne moved the marble to the box",
        "was_present": "false",
        "was_told": "false"
    }
}
~~~
