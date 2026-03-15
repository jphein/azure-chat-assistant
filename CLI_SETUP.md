# CLI Setup Guide

This server uses three cloud CLIs for dynamic model discovery. All are optional but recommended.

## Azure CLI (az)

**Install:**
```bash
# Debian/Ubuntu
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# macOS
brew install azure-cli

# Windows
winget install Microsoft.AzureCLI
```

**Login:**
```bash
az login
```

**Used for:**
- List deployable OpenAI models in your subscription
- Detect which models are already deployed

**Commands the server runs:**
```bash
az cognitiveservices account list-models \
  --name <resource-name> \
  --resource-group <resource-group>

az cognitiveservices account deployment list \
  --name <resource-name> \
  --resource-group <resource-group>
```

**Configure (optional — auto-detected from endpoint):**
```bash
# The server extracts resource name from your endpoint
# If you need to set them explicitly:
export AZURE_RESOURCE_NAME=my-ai-resource
export AZURE_RESOURCE_GROUP=my-rg
```

---

## AWS CLI (aws)

**Install:**
```bash
# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Windows
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

**Configure:**
```bash
aws configure
# Enter: Access Key ID, Secret Key, Region (us-east-1 recommended)
```

**Used for:**
- List available Bedrock foundation models
- Check model access status

**Commands the server runs:**
```bash
aws bedrock list-foundation-models --region us-east-1

aws bedrock get-foundation-model-availability \
  --model-identifier <model-id> --region us-east-1
```

---

## Google Cloud CLI (gcloud)

**Install:**
```bash
# Debian/Ubuntu
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
  sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
sudo apt update && sudo apt install google-cloud-cli

# macOS
brew install google-cloud-sdk

# Windows
# Download from: https://cloud.google.com/sdk/docs/install
```

**Configure:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-east4
```

**Used for:**
- Get OAuth tokens for Vertex AI
- List available Gemini models

**Commands the server runs:**
```bash
gcloud auth print-access-token

gcloud ai models list --region=us-east4
```

---

## Quick Status Check

Run the scan tool to test all providers:
```
Use tool: scan
```

Or test CLIs directly:
```bash
# Azure - list deployable models
az cognitiveservices account list-models --name myresource --resource-group myrg --output table

# AWS - list Bedrock models
aws bedrock list-foundation-models --region us-east-1 --query 'modelSummaries[*].modelId' --output table

# Google - list Vertex AI models
gcloud ai models list --region=us-east4 --format="table(name,displayName)"
```

## Troubleshooting

### Azure: "InvalidSubscriptionId"
```bash
az account list --output table   # List subscriptions
az account set --subscription <sub-id>   # Switch subscription
```

### AWS: "AccessDeniedException"
Your IAM user needs `bedrock:ListFoundationModels` permission. Add this policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:*"],
    "Resource": "*"
  }]
}
```

### Google: "PERMISSION_DENIED"
```bash
gcloud auth login                    # Re-authenticate
gcloud services enable aiplatform.googleapis.com   # Enable Vertex AI API
```
