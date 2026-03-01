targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment')
param environmentName string

@description('Location for all resources')
param location string

@secure()
@description('GitHub token for Copilot SDK')
param githubToken string = ''

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

var tags = { 'azd-env-name': environmentName }
var resourceSuffix = take(uniqueString(subscription().id, environmentName), 6)
var shortName = take(replace(environmentName, '-', ''), 10)

resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module resources './resources.bicep' = {
  name: 'resources'
  scope: rg
  params: {
    environmentName: environmentName
    location: location
    tags: tags
    githubToken: githubToken
    resourceSuffix: resourceSuffix
    shortName: shortName
    useAzureModel: useAzureModel
    azureModelName: azureModelName
    azureModelVersion: azureModelVersion
    azureAiFoundryResourceUrl: azureAiFoundryResourceUrl
  }
}

output AZURE_CONTAINER_APP_AGENT_URL string = resources.outputs.agentContainerAppUrl
output AZURE_CONTAINER_APP_WEB_URL string = resources.outputs.webContainerAppUrl
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = resources.outputs.registryName
output AZURE_MODEL_NAME string = useAzureModel ? azureModelName : ''
output AZURE_OPENAI_ENDPOINT string = useAzureModel ? resources.outputs.azureOpenAiEndpoint : ''
