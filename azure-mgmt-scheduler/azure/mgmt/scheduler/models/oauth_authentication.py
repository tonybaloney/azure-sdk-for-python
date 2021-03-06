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

from .http_authentication import HttpAuthentication


class OAuthAuthentication(HttpAuthentication):
    """OAuthAuthentication

    :param str type: Gets or sets the http authentication type. Possible
     values include: 'NotSpecified', 'ClientCertificate',
     'ActiveDirectoryOAuth', 'Basic'
    :param str secret: Gets or sets the secret.
    :param str tenant: Gets or sets the tenant.
    :param str audience: Gets or sets the audience.
    :param str client_id: Gets or sets the client identifier.
    """

    _required = []

    _attribute_map = {
        'secret': {'key': 'secret', 'type': 'str'},
        'tenant': {'key': 'tenant', 'type': 'str'},
        'audience': {'key': 'audience', 'type': 'str'},
        'client_id': {'key': 'clientId', 'type': 'str'},
    }

    def __init__(self, type=None, secret=None, tenant=None, audience=None, client_id=None):
        super(OAuthAuthentication, self).__init__(type=type)
        self.secret = secret
        self.tenant = tenant
        self.audience = audience
        self.client_id = client_id
