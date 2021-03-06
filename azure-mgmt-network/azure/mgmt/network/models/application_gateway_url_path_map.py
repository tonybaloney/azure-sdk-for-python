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


class ApplicationGatewayUrlPathMap(SubResource):
    """
    UrlPathMap of application gateway

    :param str id: Resource Id
    :param str name: Gets name of the resource that is unique within a
     resource group. This name can be used to access the resource
    :param str etag: A unique read-only string that changes whenever the
     resource is updated
    :param SubResource default_backend_address_pool: Gets or sets default
     backend address pool resource of URL path map
    :param SubResource default_backend_http_settings: Gets or sets default
     backend http settings resource of URL path map
    :param list path_rules: Gets or sets path rule of URL path map resource
    :param str provisioning_state: Gets or sets Provisioning state of the
     backend http settings resource Updating/Deleting/Failed
    """

    _required = []

    _attribute_map = {
        'name': {'key': 'name', 'type': 'str'},
        'etag': {'key': 'etag', 'type': 'str'},
        'default_backend_address_pool': {'key': 'properties.defaultBackendAddressPool', 'type': 'SubResource', 'flatten': True},
        'default_backend_http_settings': {'key': 'properties.defaultBackendHttpSettings', 'type': 'SubResource', 'flatten': True},
        'path_rules': {'key': 'properties.pathRules', 'type': '[ApplicationGatewayPathRule]', 'flatten': True},
        'provisioning_state': {'key': 'properties.provisioningState', 'type': 'str', 'flatten': True},
    }

    def __init__(self, id=None, name=None, etag=None, default_backend_address_pool=None, default_backend_http_settings=None, path_rules=None, provisioning_state=None):
        super(ApplicationGatewayUrlPathMap, self).__init__(id=id)
        self.name = name
        self.etag = etag
        self.default_backend_address_pool = default_backend_address_pool
        self.default_backend_http_settings = default_backend_http_settings
        self.path_rules = path_rules
        self.provisioning_state = provisioning_state
