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


class LocalNetworkGateway(Resource):
    """
    A common class for general resource information

    :param str id: Resource Id
    :param str name: Resource name
    :param str type: Resource type
    :param str location: Resource location
    :param dict tags: Resource tags
    :param str etag: Gets a unique read-only string that changes whenever the
     resource is updated
    :param AddressSpace local_network_address_space: Local network site
     Address space
    :param str gateway_ip_address: IP address of local network gateway.
    :param str resource_guid: Gets or sets resource guid property of the
     LocalNetworkGateway resource
    :param str provisioning_state: Gets or sets Provisioning state of the
     LocalNetworkGateway resource Updating/Deleting/Failed
    """

    _required = []

    _attribute_map = {
        'etag': {'key': 'etag', 'type': 'str'},
        'local_network_address_space': {'key': 'properties.localNetworkAddressSpace', 'type': 'AddressSpace', 'flatten': True},
        'gateway_ip_address': {'key': 'properties.gatewayIpAddress', 'type': 'str', 'flatten': True},
        'resource_guid': {'key': 'properties.resourceGuid', 'type': 'str', 'flatten': True},
        'provisioning_state': {'key': 'properties.provisioningState', 'type': 'str', 'flatten': True},
    }

    def __init__(self, id=None, name=None, type=None, location=None, tags=None, etag=None, local_network_address_space=None, gateway_ip_address=None, resource_guid=None, provisioning_state=None):
        super(LocalNetworkGateway, self).__init__(id=id, name=name, type=type, location=location, tags=tags)
        self.etag = etag
        self.local_network_address_space = local_network_address_space
        self.gateway_ip_address = gateway_ip_address
        self.resource_guid = resource_guid
        self.provisioning_state = provisioning_state
