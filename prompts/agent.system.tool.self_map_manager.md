## {self} Map Manager (MHH-EI)
manage user identity attachment maps for emotional intelligence
load map at session start, add items when user reveals attachments

### self_map_manager
CRUD operations on the user's {self} map (identity attachments stored in ChromaDB)
- action: load | add | update | remove | list | restore
- user_id: user identifier (auto-resolved if omitted)
- For add: label, quadrant (people|accomplishments|life_story|ideas_likes), power_level (1-10), valence (-10 to +10), source (inferred|user_stated|agent_observed)
- For update: item_id + fields to change
- For remove: item_id

Load at session start:
~~~json
{
    "thoughts": [
        "Starting session, I should load the user's {self} map for emotional awareness."
    ],
    "headline": "Loading {self} map",
    "tool_name": "self_map_manager",
    "tool_args": {
        "action": "load",
        "user_id": "smsmatt"
    }
}
~~~

Add new attachment:
~~~json
{
    "thoughts": [
        "The user mentioned their wife — this is a high-power people attachment."
    ],
    "headline": "Adding attachment to {self} map",
    "tool_name": "self_map_manager",
    "tool_args": {
        "action": "add",
        "label": "wife",
        "quadrant": "people",
        "power_level": "9",
        "valence": "8",
        "source": "user_stated"
    }
}
~~~
