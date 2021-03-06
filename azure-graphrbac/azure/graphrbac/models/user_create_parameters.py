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


class UserCreateParameters(Model):
    """
    Request parameters for create a new user

    :param str user_principal_name: User Principal Name
    :param bool account_enabled: Enable the account
    :param str display_name: User display name
    :param str mail_nickname: Mail nick name
    :param UserCreateParametersPasswordProfile password_profile: Password
     Profile
    """

    _required = ['user_principal_name', 'account_enabled', 'display_name', 'mail_nickname', 'password_profile']

    _attribute_map = {
        'user_principal_name': {'key': 'userPrincipalName', 'type': 'str'},
        'account_enabled': {'key': 'accountEnabled', 'type': 'bool'},
        'display_name': {'key': 'displayName', 'type': 'str'},
        'mail_nickname': {'key': 'mailNickname', 'type': 'str'},
        'password_profile': {'key': 'passwordProfile', 'type': 'UserCreateParametersPasswordProfile'},
    }

    def __init__(self, user_principal_name, account_enabled, display_name, mail_nickname, password_profile):
        self.user_principal_name = user_principal_name
        self.account_enabled = account_enabled
        self.display_name = display_name
        self.mail_nickname = mail_nickname
        self.password_profile = password_profile
