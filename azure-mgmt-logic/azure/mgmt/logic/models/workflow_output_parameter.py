# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft and contributors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from .workflow_parameter import WorkflowParameter


class WorkflowOutputParameter(WorkflowParameter):
    """WorkflowOutputParameter

    :param str type: Gets or sets the type. Possible values include:
     'NotSpecified', 'String', 'SecureString', 'Int', 'Float', 'Bool',
     'Array', 'Object', 'SecureObject'
    :param object value: Gets or sets the value.
    :param object metadata: Gets or sets the metadata.
    :param object error: Gets the error.
    """

    _required = []

    _attribute_map = {
        'error': {'key': 'error', 'type': 'object'},
    }

    def __init__(self, type=None, value=None, metadata=None, error=None):
        super(WorkflowOutputParameter, self).__init__(type=type, value=value, metadata=metadata)
        self.error = error
