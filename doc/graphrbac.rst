Azure Active Directory Graph Rbac API
=====================================

Create the client
-----------------

The following code creates an instance of the client.

You will need to provide your ``subscription_id`` which can be retrieved
from `your subscription list <https://manage.windowsazure.com/#Workspaces/AdminTasks/SubscriptionMapping>`__.

See :doc:`Resource Management Authentication <resourcemanagementauthentication>`
for details on getting a Credentials instance.

You will also need the tenant id of the AD you want to manage. Could be the AD UUID or domain name.
`You can follow this documentation to get it <https://msdn.microsoft.com/fr-fr/library/azure/ad/graph/howto/azure-ad-graph-api-operations-overview#TenantIdentifier>`__.

.. code:: python

    from azure.graphrbac import GraphRbacManagementClient, GraphRbacManagementClientConfiguration

    # TODO: Replace this with your subscription id
    subscription_id = '33333333-3333-3333-3333-333333333333'
    # TODO: must be an instance of 
    # - msrestazure.azure_active_directory.UserPassCredentials
    # - msrestazure.azure_active_directory.ServicePrincipalCredentials
    credentials = ...
    tenant_id = "myad.onmicrosoft.com"

    graphrbac_client = GraphRbacManagementClient(
        GraphRbacManagementClientConfiguration(
            credentials,
            subscription_id,
            tenant_id
        )
    )

Manage users
------------

The following code creates a user, get it directly and by list filtering, and then delete it.
`Filter syntax can be found here <https://msdn.microsoft.com/fr-fr/library/azure/ad/graph/howto/azure-ad-graph-api-supported-queries-filters-and-paging-options#-filter>`__.

.. code:: python

    from azure.graphrbac.models import UserCreateParameters, UserCreateParametersPasswordProfile

    user = graphrbac_client.user.create(
        UserCreateParameters(
            user_principal_name="testbuddy@{}".format(MY_AD_DOMAIN),
            account_enabled=False,
            display_name='Test Buddy',
            mail_nickname='testbuddy',
            password_profile=UserCreateParametersPasswordProfile(
                password='MyStr0ngP4ssword',
                force_change_password_next_login=True
            )
        )
    )
    # user is a User instance
    self.assertEqual(user.display_name, 'Test Buddy')

    user = graphrbac_client.user.get(user.object_id)
    self.assertEqual(user.display_name, 'Test Buddy')

    for user in graphrbac_client.user.list(filter="displayName eq 'Test Buddy'"):
        self.assertEqual(user.display_name, 'Test Buddy')

    graphrbac_client.user.delete(user.object_id)
