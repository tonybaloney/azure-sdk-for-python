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

from .sub_resource import SubResource


class WorkflowAccessKey(SubResource):
    """WorkflowAccessKey

    :param str id: Gets or sets the resource id.
    :param str name: Gets the workflow access key name.
    :param str type: Gets the workflow access key type.
    :param datetime not_before: Gets or sets the not-before time.
    :param datetime not_after: Gets or sets the not-after time.
    """

    _required = []

    _attribute_map = {
        'name': {'key': 'name', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'not_before': {'key': 'properties.notBefore', 'type': 'iso-8601', 'flatten': True},
        'not_after': {'key': 'properties.notAfter', 'type': 'iso-8601', 'flatten': True},
    }

    def __init__(self, id=None, name=None, type=None, not_before=None, not_after=None):
        super(WorkflowAccessKey, self).__init__(id=id)
        self.name = name
        self.type = type
        self.not_before = not_before
        self.not_after = not_after
