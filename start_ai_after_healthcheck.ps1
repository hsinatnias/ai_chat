# start_ai_after_healthcheck.ps1 (blocking version)
$root = "C:\ai_chatbot\rag_chat4"
$python = Join-Path $root ".venv\Scripts\python.exe"
$uvicornArgs = @('-m','uvicorn','app:app','--host','0.0.0.0','--port','8000')

# health endpoints
$ollamaUrl = "http://127.0.0.1:11434/"
$qdrantUrl = "http://127.0.0.1:6333/collections"

# timeouts
$timeoutSeconds = 120
$intervalSeconds = 2
$deadline = (Get-Date).AddSeconds($timeoutSeconds)

function Wait-ForUrl($url, $name) {
    Write-Output "Waiting for $name at $url..."
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

$ollamaReady = Wait-ForUrl $ollamaUrl "Ollama"
$qdrantReady = Wait-ForUrl $qdrantUrl "Qdrant"

if (-not ($ollamaReady -and $qdrantReady)) {
    $msg = "{0} - Health check failed: Ollama={1} Qdrant={2}" -f (Get-Date), $ollamaReady, $qdrantReady
    Add-Content -Path (Join-Path $root "logs\service_start_errors.log") -Value $msg
    # Non-zero exit so WinSW can mark as failed / retry depending on config
    exit 1
}

# Both ready -> run uvicorn in the foreground (blocking).
# This call inherits stdout/stderr so WinSW can capture logs.
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
