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

from msrest.serialization import Model


class ManagementLockObject(Model):
    """
    Management lock information.

    :param str id: Gets or sets the Id of the lock.
    :param str type: Gets or sets the type of the lock.
    :param str name: Gets or sets the name of the lock.
    :param str level: Gets or sets the lock level of the management lock.
     Possible values include: 'NotSpecified', 'CanNotDelete', 'ReadOnly'
    :param str notes: Gets or sets the notes of the management lock.
    """

    _required = []

    _attribute_map = {
        'id': {'key': 'id', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'name': {'key': 'name', 'type': 'str'},
        'level': {'key': 'properties.level', 'type': 'LockLevel', 'flatten': True},
        'notes': {'key': 'properties.notes', 'type': 'str', 'flatten': True},
    }

    def __init__(self, id=None, type=None, name=None, level=None, notes=None):
        self.id = id
        self.type = type
        self.name = name
        self.level = level
        self.notes = notes
