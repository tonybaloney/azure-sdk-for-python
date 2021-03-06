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


class StorageProfile(Model):
    """
    Describes a storage profile.

    :param ImageReference image_reference: Gets or sets the image reference.
    :param OSDisk os_disk: Gets or sets the OS disk.
    :param list data_disks: Gets or sets the data disks.
    """

    _required = []

    _attribute_map = {
        'image_reference': {'key': 'imageReference', 'type': 'ImageReference'},
        'os_disk': {'key': 'osDisk', 'type': 'OSDisk'},
        'data_disks': {'key': 'dataDisks', 'type': '[DataDisk]'},
    }

    def __init__(self, image_reference=None, os_disk=None, data_disks=None):
        self.image_reference = image_reference
        self.os_disk = os_disk
        self.data_disks = data_disks
