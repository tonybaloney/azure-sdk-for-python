﻿# coding: utf-8

#-------------------------------------------------------------------------
# Copyright (c) Microsoft.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#--------------------------------------------------------------------------
import unittest

import azure.mgmt.logic
import azure.mgmt.web
from testutils.common_recordingtestcase import record
from tests.mgmt_testcase import HttpStatusCode, AzureMgmtTestCase


class MgmtAppsTest(AzureMgmtTestCase):

    def setUp(self):
        super(MgmtAppsTest, self).setUp()
        self.logic_client = self.create_mgmt_client(
            azure.mgmt.logic.LogicManagementClientConfiguration,
            azure.mgmt.logic.LogicManagementClient
        )
        self.web_client = self.create_mgmt_client(
            azure.mgmt.web.WebSiteManagementClientConfiguration,
            azure.mgmt.web.WebSiteManagementClient
        )

    @unittest.skip("https://github.com/Azure/azure-rest-api-specs/issues/137")
    @record
    def test_logic(self):
        self.create_resource_group()

        # Create App Service Plan
        app_service_plan_name = self.get_resource_name('pyarmappserviceplan')
        app_service_plan = self.web_client.server_farms.create_or_update_server_farm(
            self.group_name,
            app_service_plan_name,
            azure.mgmt.web.models.ServerFarmWithRichSku(
                location='West US',
                sku=azure.mgmt.web.models.SkuDescription(
                    name='F1',
                    tier='Free'
                )
            )
        )
        app_service_plan = app_service_plan.result()

        workflow_name = '12HourHeartBeat'
        self.logic_client.workflows.create_or_update(
            self.group_name,
            workflow_name,
            azure.mgmt.logic.models.Workflow(
                location='West US',
                sku=azure.mgmt.logic.models.Sku(
                    name='Free',
                    plan=azure.mgmt.logic.models.ResourceReference(
                        name=app_service_plan_name,
                        type='Microsoft.Web/ServerFarms',
                        id=app_service_plan.id
                    )
                )
            )
        )


#------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
