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


class SiteLogsConfig(Resource):
    """
    Configuration of Azure web site

    :param str id: Resource Id
    :param str name: Resource Name
    :param str location: Resource Location
    :param str type: Resource type
    :param dict tags: Resource tags
    :param ApplicationLogsConfig application_logs: Application logs
     configuration
    :param HttpLogsConfig http_logs: Http logs configuration
    :param EnabledConfig failed_requests_tracing: Failed requests tracing
     configuration
    :param EnabledConfig detailed_error_messages: Detailed error messages
     configuration
    """

    _required = []

    _attribute_map = {
        'application_logs': {'key': 'properties.applicationLogs', 'type': 'ApplicationLogsConfig', 'flatten': True},
        'http_logs': {'key': 'properties.httpLogs', 'type': 'HttpLogsConfig', 'flatten': True},
        'failed_requests_tracing': {'key': 'properties.failedRequestsTracing', 'type': 'EnabledConfig', 'flatten': True},
        'detailed_error_messages': {'key': 'properties.detailedErrorMessages', 'type': 'EnabledConfig', 'flatten': True},
    }

    def __init__(self, location, id=None, name=None, type=None, tags=None, application_logs=None, http_logs=None, failed_requests_tracing=None, detailed_error_messages=None):
        super(SiteLogsConfig, self).__init__(id=id, name=name, location=location, type=type, tags=tags)
        self.application_logs = application_logs
        self.http_logs = http_logs
        self.failed_requests_tracing = failed_requests_tracing
        self.detailed_error_messages = detailed_error_messages
