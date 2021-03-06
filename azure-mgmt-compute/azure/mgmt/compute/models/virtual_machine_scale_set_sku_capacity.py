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


class VirtualMachineScaleSetSkuCapacity(Model):
    """
    Describes scaling information of a sku.

    :param long minimum: Gets the minimum capacity.
    :param long maximum: Gets the maximum capacity that can be set.
    :param long default_capacity: Gets the default capacity.
    :param str scale_type: Gets the scale type applicable to the sku.
     Possible values include: 'Automatic', 'None'
    """

    _required = []

    _attribute_map = {
        'minimum': {'key': 'minimum', 'type': 'long'},
        'maximum': {'key': 'maximum', 'type': 'long'},
        'default_capacity': {'key': 'defaultCapacity', 'type': 'long'},
        'scale_type': {'key': 'scaleType', 'type': 'VirtualMachineScaleSetSkuScaleType'},
    }

    def __init__(self, minimum=None, maximum=None, default_capacity=None, scale_type=None):
        self.minimum = minimum
        self.maximum = maximum
        self.default_capacity = default_capacity
        self.scale_type = scale_type
