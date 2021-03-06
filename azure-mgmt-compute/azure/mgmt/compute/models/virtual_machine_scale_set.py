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

from .resource import Resource


class VirtualMachineScaleSet(Resource):
    """
    Describes a Virtual Machine Scale Set.

    :param str id: Resource Id
    :param str name: Resource name
    :param str type: Resource type
    :param str location: Resource location
    :param dict tags: Resource tags
    :param Sku sku: Gets or sets the virtual machine scale set sku.
    :param UpgradePolicy upgrade_policy: Gets or sets the upgrade policy.
    :param VirtualMachineScaleSetVMProfile virtual_machine_profile: Gets or
     sets the virtual machine profile.
    :param str provisioning_state: Gets or sets the provisioning state, which
     only appears in the response.
    """

    _required = []

    _attribute_map = {
        'sku': {'key': 'sku', 'type': 'Sku'},
        'upgrade_policy': {'key': 'properties.upgradePolicy', 'type': 'UpgradePolicy', 'flatten': True},
        'virtual_machine_profile': {'key': 'properties.virtualMachineProfile', 'type': 'VirtualMachineScaleSetVMProfile', 'flatten': True},
        'provisioning_state': {'key': 'properties.provisioningState', 'type': 'str', 'flatten': True},
    }

    def __init__(self, location, id=None, name=None, type=None, tags=None, sku=None, upgrade_policy=None, virtual_machine_profile=None, provisioning_state=None):
        super(VirtualMachineScaleSet, self).__init__(id=id, name=name, type=type, location=location, tags=tags)
        self.sku = sku
        self.upgrade_policy = upgrade_policy
        self.virtual_machine_profile = virtual_machine_profile
        self.provisioning_state = provisioning_state
