targetScope = 'resourceGroup'

@description('Name of the environment')
param environmentName string

@description('Location for all resources')
param location string = resourceGroup().location

@description('Tags for all resources')
param tags object = {}

@secure()
@description('GitHub token for Copilot SDK')
param githubToken string = ''

@description('Unique suffix for resource names')
param resourceSuffix string

@description('Short environment name for constrained resources')
param shortName string

@description('Deploy Azure OpenAI for BYOM. Set to true to provision AI resources.')
param useAzureModel bool = false

@description('Azure OpenAI model deployment name (must support Copilot SDK encrypted content)')
@allowed([
  'o4-mini'
  'o3'
  'o3-mini'
  'gpt-4.1'
  'gpt-5'
  'gpt-5-mini'
  'gpt-5.1'
  'gpt-5.1-mini'
  'gpt-5.1-nano'
  'gpt-5.2-codex'
  'codex-mini'
])
param azureModelName string = 'o4-mini'

@description('Azure OpenAI model version (must match the model name)')
param azureModelVersion string = '2025-04-16'

@description('Azure AI Foundry resource URL (override when useAzureModel is true)')
param azureAiFoundryResourceUrl string = ''

// ===================== //
// AZD Pattern: Monitoring (Log Analytics + App Insights)
// ===================== //

module monitoring 'br/public:avm/ptn/azd/monitoring:0.2.1' = {
  name: 'monitoring'
  params: {
    logAnalyticsName: 'law-${environmentName}-${resourceSuffix}'
    applicationInsightsName: 'ai-${environmentName}-${resourceSuffix}'
    location: location
    tags: tags
  }
}

// ===================== //
// AVM Resource: Managed Identity
// ===================== //

module managedIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.5.0' = {
  name: 'managed-identity'
  params: {
    name: 'id-${environmentName}-${resourceSuffix}'
    location: location
    tags: tags
  }
}

// ===================== //
// AVM Resource: Key Vault (stores GITHUB_TOKEN)
// ===================== //

module keyVault 'br/public:avm/res/key-vault/vault:0.13.3' = {
  name: 'key-vault'
  params: {
    name: 'kv-${shortName}-${resourceSuffix}'
    location: location
    tags: tags
    enableRbacAuthorization: true
    enableSoftDelete: true
    enablePurgeProtection: false
    softDeleteRetentionInDays: 7
    sku: 'standard'
    secrets: [
      {
        name: 'github-token'
        value: githubToken
      }
    ]
    roleAssignments: [
      {
        principalId: managedIdentity.outputs.principalId
        roleDefinitionIdOrName: 'Key Vault Secrets User'
        principalType: 'ServicePrincipal'
      }
    ]
  }
}

// ===================== //
// AZD Pattern: Container Apps Stack (Environment + ACR)
// ===================== //

module containerAppsStack 'br/public:avm/ptn/azd/container-apps-stack:0.3.0' = {
  name: 'container-apps-stack'
  params: {
    containerAppsEnvironmentName: 'cae-${environmentName}-${resourceSuffix}'
    containerRegistryName: 'acr${shortName}${resourceSuffix}'
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
    appInsightsConnectionString: monitoring.outputs.applicationInsightsConnectionString
    location: location
    tags: tags
    acrSku: 'Basic'
    acrAdminUserEnabled: false
    zoneRedundant: false
    publicNetworkAccess: 'Enabled'
  }
}

// ===================== //
// Azure OpenAI (conditional, for BYOM)
// ===================== //

resource openai 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (useAzureModel) {
  name: 'oai-${environmentName}-${resourceSuffix}'
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: 'oai-${environmentName}-${resourceSuffix}'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
  }
}

resource openaiDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (useAzureModel) {
  parent: openai
  name: azureModelName
  sku: {
    name: 'GlobalStandard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: azureModelName
      version: azureModelVersion
    }
  }
}

resource openaiRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (useAzureModel) {
  scope: openai
  name: guid(resourceGroup().id, 'openai-role', 'id-${environmentName}-${resourceSuffix}', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: managedIdentity.outputs.principalId
    principalType: 'ServicePrincipal'
  }
}

// ===================== //
// AZD Pattern: ACR Container App – Copilot Agent (internal)
// ===================== //

module containerAppAgent 'br/public:avm/ptn/azd/acr-container-app:0.4.0' = {
  name: 'container-app-agent'
  params: {
    name: 'ca-agent-${environmentName}-${resourceSuffix}'
    location: location
    tags: union(tags, { 'azd-service-name': 'copilot-agent' })
    containerAppsEnvironmentName: containerAppsStack.outputs.environmentName
    containerRegistryName: containerAppsStack.outputs.registryName
    identityType: 'UserAssigned'
    identityName: managedIdentity.outputs.name
    userAssignedIdentityResourceId: managedIdentity.outputs.resourceId
    principalId: managedIdentity.outputs.principalId
    targetPort: 8088
    external: false
    ingressTransport: 'http'
    containerCpuCoreCount: '1.0'
    containerMemory: '2.0Gi'
    containerMinReplicas: 1
    containerMaxReplicas: 5
    env: union(
      [
        { name: 'GITHUB_TOKEN', secretRef: 'github-token' }
        { name: 'AZURE_CLIENT_ID', value: managedIdentity.outputs.clientId }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: monitoring.outputs.applicationInsightsConnectionString
        }
      ],
      useAzureModel ? [
        {
          name: 'AZURE_AI_FOUNDRY_RESOURCE_URL'
          value: !empty(azureAiFoundryResourceUrl) ? azureAiFoundryResourceUrl : openai!.properties.endpoint
        }
        { name: 'COPILOT_MODEL', value: azureModelName }
      ] : []
    )
    secrets: [
      {
        name: 'github-token'
        keyVaultUrl: keyVault.outputs.secrets[0].uri
        identity: managedIdentity.outputs.resourceId
      }
    ]
  }
}

// ===================== //
// AZD Pattern: ACR Container App – Web (external)
// ===================== //

module containerAppWeb 'br/public:avm/ptn/azd/acr-container-app:0.4.0' = {
  name: 'container-app-web'
  params: {
    name: 'ca-web-${environmentName}-${resourceSuffix}'
    location: location
    tags: union(tags, { 'azd-service-name': 'web' })
    containerAppsEnvironmentName: containerAppsStack.outputs.environmentName
    containerRegistryName: containerAppsStack.outputs.registryName
    identityType: 'UserAssigned'
    identityName: managedIdentity.outputs.name
    userAssignedIdentityResourceId: managedIdentity.outputs.resourceId
    principalId: managedIdentity.outputs.principalId
    targetPort: 80
    external: true
    ingressTransport: 'auto'
    containerCpuCoreCount: '0.25'
    containerMemory: '0.5Gi'
    containerMinReplicas: 1
    containerMaxReplicas: 3
    env: [
      {
        name: 'API_URL'
        value: 'http://${containerAppAgent.outputs.name}.internal.${containerAppsStack.outputs.defaultDomain}'
      }
    ]
  }
}

// ===================== //
// Outputs
// ===================== //

output agentContainerAppUrl string = containerAppAgent.outputs.uri
output webContainerAppUrl string = containerAppWeb.outputs.uri
output registryLoginServer string = containerAppsStack.outputs.registryLoginServer
output registryName string = containerAppsStack.outputs.registryName
output azureOpenAiEndpoint string = useAzureModel ? openai!.properties.endpoint : ''
