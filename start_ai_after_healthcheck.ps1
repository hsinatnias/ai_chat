# start_ai_after_healthcheck.ps1 (improved)
$root = "C:\ai_chatbot\rag_chat4"
$python = Join-Path $root ".venv\Scripts\python.exe"
$uvicornArgs = @('-m','uvicorn','app:app','--host','0.0.0.0','--port','8000','--reload')

# health endpoints
$ollamaModelsUrl = "http://127.0.0.1:11434/v1/models"
$qdrantUrl = "http://127.0.0.1:6333/collections"

# timeouts (per-service)
$timeoutSeconds = 120
$intervalSeconds = 2

# Ollama model to ensure loaded (optional)
$ollamaModel = $env:OLLAMA_MODEL
if (-not $ollamaModel -or $ollamaModel -eq "") { $ollamaModel = "gemma3:4b" }

function Wait-ForUrl($url, $name, $timeoutSec) {
    Write-Output "Waiting for $name at $url (timeout ${timeoutSec}s)..."
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 3 -ErrorAction Stop
            Write-Output "$name responded."
            return $true
        } catch {
            Start-Sleep -Seconds $intervalSeconds
        }
    }
    Write-Output "Timeout waiting for $name ($url)."
    return $false
}

# Wait for Qdrant (ensure service listening)
$qdrantReady = Wait-ForUrl $qdrantUrl "Qdrant" $timeoutSeconds

# Wait for Ollama service
$ollamaReady = Wait-ForUrl $ollamaModelsUrl "Ollama" $timeoutSeconds

# If Ollama up, wait until the model is visible in /v1/models
$ollamaModelReady = $false
if ($ollamaReady) {
    $modelDeadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $modelDeadline) {
        try {
            $r = Invoke-RestMethod -Uri $ollamaModelsUrl -TimeoutSec 5 -ErrorAction Stop
            # $r.data is expected to be list of {id: 'modelname'}
            if ($r -and $r.data) {
                $ids = $r.data | ForEach-Object { $_.id }
                if ($ids -contains $ollamaModel) {
                    Write-Output "Ollama model '$ollamaModel' present."
                    $ollamaModelReady = $true
                    break
                } else {
                    Write-Output "Ollama reachable but model '$ollamaModel' not listed yet. Models: $($ids -join ', ')"
                }
            }
        } catch {
            # ignore; continue polling
        }
        Start-Sleep -Seconds $intervalSeconds
    }

    # If model not present, try a gentle 'ollama pull' if the CLI exists (best-effort download)
    if (-not $ollamaModelReady) {
        $ollamaExe = (Get-Command "ollama" -ErrorAction SilentlyContinue).Source
        if ($ollamaExe) {
            Write-Output "Attempting 'ollama pull $ollamaModel' to ensure model is available..."
            try {
                $proc = Start-Process -FilePath $ollamaExe -ArgumentList @("pull", $ollamaModel) -NoNewWindow -Wait -PassThru -ErrorAction Stop
                Write-Output "'ollama pull' exit code: $($proc.ExitCode)"
            } catch {
                Write-Output "ollama pull failed (continuing): $_"
            }
            # re-check
            try {
                $r2 = Invoke-RestMethod -Uri $ollamaModelsUrl -TimeoutSec 5 -ErrorAction Stop
                $ids2 = $r2.data | ForEach-Object { $_.id }
                if ($ids2 -contains $ollamaModel) {
                    $ollamaModelReady = $true
                    Write-Output "Model '$ollamaModel' now present after pull."
                }
            } catch {
                # ignore
            }
        } else {
            Write-Output "ollama CLI not found; skipping pull."
        }
    }
}

if (-not ($ollamaReady -and $qdrantReady -and $ollamaModelReady)) {
    $msg = "{0} - Health check failed: OllamaService={1} QdrantService={2} OllamaModelLoaded={3}" -f (Get-Date), $ollamaReady, $qdrantReady, $ollamaModelReady
    $logpath = (Join-Path $root "logs\service_start_errors.log")
    Add-Content -Path $logpath -Value $msg
    Write-Output $msg
    # Non-zero exit so WinSW can mark as failed / retry depending on config
    exit 1
}

# Both ready -> run uvicorn in the foreground (blocking).
Set-Location $root
Write-Output "Starting uvicorn with $python $($uvicornArgs -join ' ')"
try {
    & $python $uvicornArgs
    $rc = $LASTEXITCODE
} catch {
    Write-Output "uvicorn failed to start: $_"
    $rc = 1
}

exit $rc
