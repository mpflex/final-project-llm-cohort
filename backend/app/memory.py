"""
TacoLLM — Session Memory

Stores per-session user constraint preferences in memory.
Preferences persist for the lifetime of the server process;
they are cleared on explicit user action or server restart.
"""

from typing import Any, Dict


class SessionMemory:
    """
    Key-value preference store keyed by session_id.

    Only truthy values are stored — False and None are silently dropped
    so that the absence of a key means "no preference" throughout the
    rest of the pipeline.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return a copy of stored preferences for the session."""
        return dict(self._store.get(session_id, {}))

    def update(self, session_id: str, constraints: Dict[str, Any]) -> None:
        """
        Merge new constraints into the session store.
        False and None values are ignored.
        """
        if session_id not in self._store:
            self._store[session_id] = {}
        for key, value in constraints.items():
            if value is not None and value is not False:
                self._store[session_id][key] = value

    def clear(self, session_id: str) -> None:
        """Remove all stored preferences for the session."""
        self._store.pop(session_id, None)
