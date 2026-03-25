## Emotional Analysis (MHH-EI Webb Equation)
analyze emotional content using the Webb Equation of Emotion EP ∆ P = ER
call when user expresses or implies emotion

### emotional_analysis
Computes the Webb Equation of Emotion for a perception event
- perception: what happened or was said (required)
- user_id: auto-resolved if omitted
- affected_item_label: which {self} map item is affected
- valence_shift: "positive" or "negative"
- accepted: has the shift been internalized? (true/false)
- source: "internal" (self-thought), "external" (outside event/person), "value" (the attachment itself)
- time_frame: "past", "now", "future"
- perspective: "internal" (how I see myself) or "external" (how others see me)
- p_weight: perception seriousness 1-10
- Special flags: is_norm_violation, is_extended, resolves_previous, is_surprise

usage:
~~~json
{
    "thoughts": [
        "The user said they're worried about losing their job. This is a fear/worry scenario.",
        "Their job is likely on their {self} map as an accomplishment with moderate-high power."
    ],
    "headline": "Analyzing emotional content",
    "tool_name": "emotional_analysis",
    "tool_args": {
        "perception": "User expressed worry about potential job loss",
        "affected_item_label": "my job",
        "valence_shift": "negative",
        "accepted": "false",
        "source": "external",
        "time_frame": "future",
        "p_weight": "6"
    }
}
~~~
