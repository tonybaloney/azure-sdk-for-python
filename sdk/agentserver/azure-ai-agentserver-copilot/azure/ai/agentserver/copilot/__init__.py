# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from typing import Optional

from copilot import SessionConfig

from ._version import VERSION


def from_copilot(session_config: Optional[SessionConfig] = None):
    """Create a CopilotAdapter agent server from a Copilot SessionConfig.

    :param session_config: Configuration for the Copilot session (model, etc.).
    :type session_config: Optional[SessionConfig]
    :return: A CopilotAdapter instance ready to run.
    :rtype: CopilotAdapter
    """
    from .copilot_adapter import CopilotAdapter

    return CopilotAdapter(session_config=session_config)


__all__ = ["from_copilot"]
__version__ = VERSION
