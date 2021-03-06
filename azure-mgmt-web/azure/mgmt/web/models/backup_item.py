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


class BackupItem(Resource):
    """
    Backup description

    :param str id: Resource Id
    :param str name: Resource Name
    :param str location: Resource Location
    :param str type: Resource type
    :param dict tags: Resource tags
    :param str storage_account_url: SAS URL for the storage account container
     which contains this backup
    :param str blob_name: Name of the blob which contains data for this backup
    :param str backup_item_name: Name of this backup
    :param str status: Backup status. Possible values include: 'InProgress',
     'Failed', 'Succeeded', 'TimedOut', 'Created', 'Skipped',
     'PartiallySucceeded', 'DeleteInProgress', 'DeleteFailed', 'Deleted'
    :param long size_in_bytes: Size of the backup in bytes
    :param datetime created: Timestamp of the backup creation
    :param str log: Details regarding this backup. Might contain an error
     message.
    :param list databases: List of databases included in the backup
    :param bool scheduled: True if this backup has been created due to a
     schedule being triggered.
    :param datetime last_restore_time_stamp: Timestamp of a last restore
     operation which used this backup.
    :param datetime finished_time_stamp: Timestamp when this backup finished.
    :param str correlation_id: Unique correlation identifier. Please use this
     along with the timestamp while communicating with Azure support.
    :param long website_size_in_bytes: Size of the original web app which has
     been backed up
    """

    _required = []

    _attribute_map = {
        'storage_account_url': {'key': 'properties.storageAccountUrl', 'type': 'str', 'flatten': True},
        'blob_name': {'key': 'properties.blobName', 'type': 'str', 'flatten': True},
        'backup_item_name': {'key': 'properties.name', 'type': 'str', 'flatten': True},
        'status': {'key': 'properties.status', 'type': 'BackupItemStatus', 'flatten': True},
        'size_in_bytes': {'key': 'properties.sizeInBytes', 'type': 'long', 'flatten': True},
        'created': {'key': 'properties.created', 'type': 'iso-8601', 'flatten': True},
        'log': {'key': 'properties.log', 'type': 'str', 'flatten': True},
        'databases': {'key': 'properties.databases', 'type': '[DatabaseBackupSetting]', 'flatten': True},
        'scheduled': {'key': 'properties.scheduled', 'type': 'bool', 'flatten': True},
        'last_restore_time_stamp': {'key': 'properties.lastRestoreTimeStamp', 'type': 'iso-8601', 'flatten': True},
        'finished_time_stamp': {'key': 'properties.finishedTimeStamp', 'type': 'iso-8601', 'flatten': True},
        'correlation_id': {'key': 'properties.correlationId', 'type': 'str', 'flatten': True},
        'website_size_in_bytes': {'key': 'properties.websiteSizeInBytes', 'type': 'long', 'flatten': True},
    }

    def __init__(self, location, id=None, name=None, type=None, tags=None, storage_account_url=None, blob_name=None, backup_item_name=None, status=None, size_in_bytes=None, created=None, log=None, databases=None, scheduled=None, last_restore_time_stamp=None, finished_time_stamp=None, correlation_id=None, website_size_in_bytes=None):
        super(BackupItem, self).__init__(id=id, name=name, location=location, type=type, tags=tags)
        self.storage_account_url = storage_account_url
        self.blob_name = blob_name
        self.backup_item_name = backup_item_name
        self.status = status
        self.size_in_bytes = size_in_bytes
        self.created = created
        self.log = log
        self.databases = databases
        self.scheduled = scheduled
        self.last_restore_time_stamp = last_restore_time_stamp
        self.finished_time_stamp = finished_time_stamp
        self.correlation_id = correlation_id
        self.website_size_in_bytes = website_size_in_bytes
