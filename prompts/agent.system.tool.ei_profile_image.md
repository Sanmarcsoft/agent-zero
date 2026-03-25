## EI Profile Image Generator (MHH-EI)
generate a text prompt for agent avatar from emotional data
call on demand or on first session with new user

### ei_profile_image
Synthesizes {self} map + emotion logs + agent personality into image generation prompt
- user_id: auto-resolved if omitted
- style: image style override (default: "digital art portrait, professional, modern")

usage:
~~~json
{
    "thoughts": [
        "The user wants to see what their agent avatar looks like based on emotional profile."
    ],
    "headline": "Generating profile image prompt",
    "tool_name": "ei_profile_image",
    "tool_args": {
        "user_id": "smsmatt",
        "style": "watercolor portrait, warm, artistic"
    }
}
~~~
