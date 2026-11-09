param([string]$Repository = 'jordankzf/vocal-loom')
$ErrorActionPreference = 'Stop'
$metadataPath = Join-Path $PSScriptRoot '..\.github\repository.json'
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
gh repo edit $Repository --description $metadata.description
if ($LASTEXITCODE -ne 0) { throw 'Could not update the repository description.' }
foreach ($topic in $metadata.topics) {
    gh repo edit $Repository --add-topic $topic
    if ($LASTEXITCODE -ne 0) { throw "Could not apply topic: $topic" }
}
Write-Output "Updated description and topics for $Repository"
