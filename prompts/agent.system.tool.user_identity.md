## User Identity Manager (MHH-EI)
manage user accounts and auth config in the ei-identity git repo
handles registration lookup updates and auth configuration

### user_identity_manager
Actions: register | get | update | list | deactivate | get_auth_config
- user_id: user identifier (required for most actions)
- For register: display_name

Register user:
~~~json
{
    "thoughts": [
        "New user needs to be registered in the identity system."
    ],
    "headline": "Registering new user",
    "tool_name": "user_identity_manager",
    "tool_args": {
        "action": "register",
        "user_id": "newuser",
        "display_name": "New User"
    }
}
~~~
