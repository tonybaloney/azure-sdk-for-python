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

from .tracked_resource import TrackedResource


class Endpoint(TrackedResource):
    """
    CDN Endpoint is the entity within a CDN Profile containing configuration
    information regarding caching behaviors and origins. The CDN Endpoint is
    exposed using the URL format <endpointname>.azureedge.net by default, but
    custom domains can also be created.

    :param str id: Resource Id
    :param str name: Resource Name
    :param str type: Resource type
    :param str location: Resource location
    :param dict tags: Resource tags
    :param str host_name: The host name of the endpoint
     {endpointName}.{DNSZone}
    :param str origin_host_header: The host header CDN provider will send
     along with content requests to origins. The default value would be the
     host name of the origin.
    :param str origin_path: The path used for origin requests
    :param list content_types_to_compress: List of content types on which
     compression will be applied. The value for the elements should be
     Internet media type.
    :param bool is_compression_enabled: Indicates whether the compression is
     enabled. Default value is false. If compression is enabled, the content
     transferred from cdn endpoint to end user will be compressed. The
     requested content must be larger than 1 byte and smaller than 1 MB.
    :param bool is_http_allowed: Indicates whether http traffic is allowed on
     the endpoint. Default value is true. At least one protocol (http or
     https) must be allowed.
    :param bool is_https_allowed: Indicates whether https traffic is allowed
     on the endpoint. Default value is true. At least one protocol (http or
     https) must be allowed.
    :param str query_string_caching_behavior: Defines the query string
     caching behavior. Possible values include: 'IgnoreQueryString',
     'BypassCaching', 'UseQueryString', 'NotSet'
    :param list origins: The set of origins of the CDN endpoint. When
     multiple origins exist, the first origin will be used as primary and
     rest will be used as failover options.
    :param str resource_state: Resource status of the endpoint. Possible
     values include: 'Creating', 'Deleting', 'Running', 'Starting',
     'Stopped', 'Stopping'
    :param str provisioning_state: Provisioning status of the endpoint.
     Possible values include: 'Creating', 'Succeeded', 'Failed'
    """

    _required = []

    _attribute_map = {
        'host_name': {'key': 'properties.hostName', 'type': 'str', 'flatten': True},
        'origin_host_header': {'key': 'properties.originHostHeader', 'type': 'str', 'flatten': True},
        'origin_path': {'key': 'properties.originPath', 'type': 'str', 'flatten': True},
        'content_types_to_compress': {'key': 'properties.contentTypesToCompress', 'type': '[str]', 'flatten': True},
        'is_compression_enabled': {'key': 'properties.isCompressionEnabled', 'type': 'bool', 'flatten': True},
        'is_http_allowed': {'key': 'properties.isHttpAllowed', 'type': 'bool', 'flatten': True},
        'is_https_allowed': {'key': 'properties.isHttpsAllowed', 'type': 'bool', 'flatten': True},
        'query_string_caching_behavior': {'key': 'properties.queryStringCachingBehavior', 'type': 'QueryStringCachingBehavior', 'flatten': True},
        'origins': {'key': 'properties.origins', 'type': '[DeepCreatedOrigin]', 'flatten': True},
        'resource_state': {'key': 'properties.resourceState', 'type': 'EndpointResourceState', 'flatten': True},
        'provisioning_state': {'key': 'properties.provisioningState', 'type': 'ProvisioningState', 'flatten': True},
    }

    def __init__(self, location, tags, id=None, name=None, type=None, host_name=None, origin_host_header=None, origin_path=None, content_types_to_compress=None, is_compression_enabled=None, is_http_allowed=None, is_https_allowed=None, query_string_caching_behavior=None, origins=None, resource_state=None, provisioning_state=None):
        super(Endpoint, self).__init__(id=id, name=name, type=type, location=location, tags=tags)
        self.host_name = host_name
        self.origin_host_header = origin_host_header
        self.origin_path = origin_path
        self.content_types_to_compress = content_types_to_compress
        self.is_compression_enabled = is_compression_enabled
        self.is_http_allowed = is_http_allowed
        self.is_https_allowed = is_https_allowed
        self.query_string_caching_behavior = query_string_caching_behavior
        self.origins = origins
        self.resource_state = resource_state
        self.provisioning_state = provisioning_state
