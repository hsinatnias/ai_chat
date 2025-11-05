# Dev bootstrap for Windows PowerShell 5.1+
param([switch]$Force)

Write-Host "Creating Python venv…" -ForegroundColor Cyan
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing requirements…" -ForegroundColor Cyan
pip install -r requirements.txt

if (!(Test-Path ".env") -or $Force) {
  Write-Host "Creating .env from example…" -ForegroundColor Cyan
  Copy-Item .env.example .env -Force
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1) Install Ollama (Windows) and pull a model:  ollama pull qwen2.5:7b"
Write-Host "  2) Start Qdrant on 127.0.0.1:6333"
Write-Host "  3) Ingest sample docs:  python ingest.py"
Write-Host "  4) Run API:            uvicorn app:app --host 127.0.0.1 --port 8000"
Write-Host "  5) Open public\chat.html and test."
