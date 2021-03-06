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


class PublicIPAddress(Resource):
    """
    PublicIPAddress resource

    :param str id: Resource Id
    :param str name: Resource name
    :param str type: Resource type
    :param str location: Resource location
    :param dict tags: Resource tags
    :param str etag: Gets a unique read-only string that changes whenever the
     resource is updated
    :param str public_ip_allocation_method: Gets or sets PublicIP allocation
     method (Static/Dynamic). Possible values include: 'Static', 'Dynamic'
    :param IPConfiguration ip_configuration:
    :param PublicIPAddressDnsSettings dns_settings: Gets or sets FQDN of the
     DNS record associated with the public IP address
    :param str ip_address:
    :param int idle_timeout_in_minutes: Gets or sets the Idletimeout of the
     public IP address
    :param str resource_guid: Gets or sets resource guid property of the
     PublicIP resource
    :param str provisioning_state: Gets or sets Provisioning state of the
     PublicIP resource Updating/Deleting/Failed
    """

    _required = []

    _attribute_map = {
        'etag': {'key': 'etag', 'type': 'str'},
        'public_ip_allocation_method': {'key': 'properties.publicIPAllocationMethod', 'type': 'IPAllocationMethod', 'flatten': True},
        'ip_configuration': {'key': 'properties.ipConfiguration', 'type': 'IPConfiguration', 'flatten': True},
        'dns_settings': {'key': 'properties.dnsSettings', 'type': 'PublicIPAddressDnsSettings', 'flatten': True},
        'ip_address': {'key': 'properties.ipAddress', 'type': 'str', 'flatten': True},
        'idle_timeout_in_minutes': {'key': 'properties.idleTimeoutInMinutes', 'type': 'int', 'flatten': True},
        'resource_guid': {'key': 'properties.resourceGuid', 'type': 'str', 'flatten': True},
        'provisioning_state': {'key': 'properties.provisioningState', 'type': 'str', 'flatten': True},
    }

    def __init__(self, id=None, name=None, type=None, location=None, tags=None, etag=None, public_ip_allocation_method=None, ip_configuration=None, dns_settings=None, ip_address=None, idle_timeout_in_minutes=None, resource_guid=None, provisioning_state=None):
        super(PublicIPAddress, self).__init__(id=id, name=name, type=type, location=location, tags=tags)
        self.etag = etag
        self.public_ip_allocation_method = public_ip_allocation_method
        self.ip_configuration = ip_configuration
        self.dns_settings = dns_settings
        self.ip_address = ip_address
        self.idle_timeout_in_minutes = idle_timeout_in_minutes
        self.resource_guid = resource_guid
        self.provisioning_state = provisioning_state
