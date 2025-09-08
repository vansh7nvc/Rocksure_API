# Helper script to init, commit, add remote and push
param(
    [string]$RemoteUrl = 'https://github.com/vansh7nvc/vansh7nvc.git'
)

$log = Join-Path $PSScriptRoot 'push_log.txt'
function Log { param($m) Add-Content -Path $log -Value $m }
Remove-Item -Path $log -ErrorAction SilentlyContinue

Log "PWD: $(Get-Location)"
Log "Git version: $(git --version 2>&1)"

$inside = git rev-parse --is-inside-work-tree 2>&1
Log "InsideRepo: $inside"
if ($LASTEXITCODE -ne 0) {
    Log "Initializing git repo..."
    git init 2>&1 | ForEach-Object { Log $_ }
}

Log "Configuring user..."
git config user.email 'devnull@example.com' 2>&1 | ForEach-Object { Log $_ }
git config user.name 'Repo Bot' 2>&1 | ForEach-Object { Log $_ }

Log "Staging files..."
git add . 2>&1 | ForEach-Object { Log $_ }

Log "Committing..."
git commit -m 'Initial commit: add FastAPI app and requirements' 2>&1 | ForEach-Object { Log $_ }

Log "Setting remote to $RemoteUrl"
# remove origin if exists
git remote remove origin 2>&1 | ForEach-Object { Log $_ }
git remote add origin $RemoteUrl 2>&1 | ForEach-Object { Log $_ }

Log "Setting branch to main"
git branch -M main 2>&1 | ForEach-Object { Log $_ }

Log "Pushing to remote (this may prompt for credentials)..."
git push -u origin main 2>&1 | ForEach-Object { Log $_ }

Log "--- END ---"
Write-Output "Wrote log to $log"
