# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import os
from typing import Awaitable, Callable, Optional, Union

from copilot import SessionConfig
from copilot.types import PermissionRequest, PermissionRequestResult

from ._version import VERSION

# Public type alias so callers can annotate their callback without importing
# internal modules.
PermissionHandlerFn = Callable[
    [PermissionRequest, dict[str, str]],
    Union[PermissionRequestResult, Awaitable[PermissionRequestResult]],
]


def from_copilot(
    session_config: Optional[SessionConfig] = None,
    *,
    acl_path: Optional[Union[str, os.PathLike]] = None,
    on_permission_request: Optional[PermissionHandlerFn] = None,
):
    """Create a CopilotAdapter agent server from a Copilot SessionConfig.

    Tool Permission Handling
    ------------------------
    The Copilot SDK **denies all tool calls by default** unless a permission
    handler is provided.  You must supply one of the following:

    * **YAML ACL file** — pass *acl_path* (or set the ``TOOL_ACL_PATH``
      environment variable) to load a declarative rule-set from a YAML file.
    * **Callback function** — pass *on_permission_request* to implement
      permission logic in Python directly.

    If both *acl_path* and *on_permission_request* are provided, the YAML
    ACL takes priority.

    See :class:`~azure.ai.agentserver.copilot.tool_acl.ToolAcl` and the
    ``samples/hosted_agent/tools_acl.yaml`` example for the YAML schema.

    :param session_config: Configuration for the Copilot session (model etc.).
    :type session_config: Optional[SessionConfig]
    :param acl_path: Path to a YAML tool ACL file.  Takes priority over the
        ``TOOL_ACL_PATH`` environment variable and *on_permission_request*.
    :type acl_path: Optional[str | os.PathLike]
    :param on_permission_request: Callback invoked for every tool permission
        request.  Receives a ``PermissionRequest`` dict and a context dict;
        must return a ``PermissionRequestResult``.  Can be sync or async.
    :type on_permission_request: Optional[PermissionHandlerFn]
    :return: A CopilotAdapter instance ready to run.
    :rtype: CopilotAdapter
    """
    from .copilot_adapter import CopilotAdapter
    from .tool_acl import ToolAcl

    acl = ToolAcl.from_file(acl_path) if acl_path is not None else None
    return CopilotAdapter(
        session_config=session_config,
        acl=acl,
        on_permission_request=on_permission_request,
    )


__all__ = ["from_copilot", "PermissionHandlerFn", "ToolAcl"]
__version__ = VERSION
