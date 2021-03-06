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


class SourceControl(Resource):
    """
    Describes the Source Control OAuth Token

    :param str id: Resource Id
    :param str name: Resource Name
    :param str location: Resource Location
    :param str type: Resource type
    :param dict tags: Resource tags
    :param str source_control_name: Name or Source Control Type
    :param str token: OAuth Access Token
    :param str token_secret: OAuth Access Token Secret
    :param str refresh_token: OAuth Refresh Token
    :param datetime expiration_time: OAuth Token Expiration
    """

    _required = []

    _attribute_map = {
        'source_control_name': {'key': 'properties.name', 'type': 'str', 'flatten': True},
        'token': {'key': 'properties.token', 'type': 'str', 'flatten': True},
        'token_secret': {'key': 'properties.tokenSecret', 'type': 'str', 'flatten': True},
        'refresh_token': {'key': 'properties.refreshToken', 'type': 'str', 'flatten': True},
        'expiration_time': {'key': 'properties.expirationTime', 'type': 'iso-8601', 'flatten': True},
    }

    def __init__(self, location, id=None, name=None, type=None, tags=None, source_control_name=None, token=None, token_secret=None, refresh_token=None, expiration_time=None):
        super(SourceControl, self).__init__(id=id, name=name, location=location, type=type, tags=tags)
        self.source_control_name = source_control_name
        self.token = token
        self.token_secret = token_secret
        self.refresh_token = refresh_token
        self.expiration_time = expiration_time
