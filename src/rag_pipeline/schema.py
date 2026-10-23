"""Response schema shared by all providers."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "thoughts": {"type": "string"},
    },
    "required": ["message", "thoughts"],
    "additionalProperties": False,
}
