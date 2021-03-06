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


class WorkerPool(Resource):
    """
    Worker pool of a hostingEnvironment (App Service Environment)

    :param str id: Resource Id
    :param str name: Resource Name
    :param str location: Resource Location
    :param str type: Resource type
    :param dict tags: Resource tags
    :param SkuDescription sku:
    :param int worker_size_id: Worker size id for referencing this worker pool
    :param str compute_mode: Shared or dedicated web app hosting. Possible
     values include: 'Shared', 'Dedicated'
    :param str worker_size: VM size of the worker pool instances
    :param int worker_count: Number of instances in the worker pool
    :param list instance_names: Names of all instances in the worker pool
     (read only)
    """

    _required = []

    _attribute_map = {
        'sku': {'key': 'sku', 'type': 'SkuDescription'},
        'worker_size_id': {'key': 'properties.workerSizeId', 'type': 'int', 'flatten': True},
        'compute_mode': {'key': 'properties.computeMode', 'type': 'ComputeModeOptions', 'flatten': True},
        'worker_size': {'key': 'properties.workerSize', 'type': 'str', 'flatten': True},
        'worker_count': {'key': 'properties.workerCount', 'type': 'int', 'flatten': True},
        'instance_names': {'key': 'properties.instanceNames', 'type': '[str]', 'flatten': True},
    }

    def __init__(self, location, id=None, name=None, type=None, tags=None, sku=None, worker_size_id=None, compute_mode=None, worker_size=None, worker_count=None, instance_names=None):
        super(WorkerPool, self).__init__(id=id, name=name, location=location, type=type, tags=tags)
        self.sku = sku
        self.worker_size_id = worker_size_id
        self.compute_mode = compute_mode
        self.worker_size = worker_size
        self.worker_count = worker_count
        self.instance_names = instance_names
