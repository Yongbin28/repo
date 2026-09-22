# =============================================================================
# WaferPulse — Build Slim Cloud Deploy Zip (run before uploading to AWS EC2)
# =============================================================================
# This script stages ONLY what the cloud server needs:
#   INCLUDED:
#     - Application source code (*.py, waferpulse/)
#     - Pre-trained models and datasets (dataset/, EquipmentData/)
#     - Configuration & user credentials (config/)
#     - Dependencies manifest (requirements.txt, pyproject.toml)
#     - Deployment scripts (deploy/)
#   EXCLUDED:
#     - Large video/presentation files (*.mp4, *.pdf, *.pptx, *.docx)
#     - Virtual environments (.venv/, fyp_env/)
#     - Cache and temp files (__pycache__, .git, *.log, build/, temp_log.txt)
#     - Unrelated reference archives
#
# Run:   powershell -ExecutionPolicy Bypass -File deploy\make_deploy_zip.ps1
# Then:  scp -i <key.pem> waferpulse_deploy.zip ubuntu@<EC2-PUBLIC-IP>:~
# =============================================================================

$proj  = Split-Path -Parent $PSScriptRoot
$stage = Join-Path (Split-Path -Parent $proj) "_waferpulse_stage"
$out   = Join-Path (Split-Path -Parent $proj) "waferpulse_deploy.zip"

Write-Host "== Staging slim copy of WaferPulse for cloud deployment ==" -ForegroundColor Cyan
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }

$rcArgs = @(
    $proj, (Join-Path $stage "waferpulse_app"),
    "/MIR", "/NFL", "/NDL", "/NJH", "/NJS",
    "/XD", "__pycache__", ".pytest_cache", ".git", ".venv", "fyp_env", "build",
           "catboost_info", "ms-playwright", "scratch", "tmp", "output", "outputs",
           "HydroSight_SourceCode_Toh Yi Jing_TP065288", ".playwright-cli",
           "seagate_softsensing", "phm2018_ion_mill", "sput_rtp",
    "/XF", "*.log", "*.mp4", "*.pdf", "*.docx", "temp_log.txt", "*.zip"
)

$null = & robocopy @rcArgs
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

# Clear Windows read-only attributes so Linux can write permissions smoothly
attrib -R (Join-Path $stage "*") /S /D

if (Test-Path $out) { Remove-Item $out -Force }

# Python zipfile writes standard forward-slash entries compatible with Linux unzip
$base = $out -replace '\.zip$', ''
python -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2])" $base $stage
Remove-Item $stage -Recurse -Force

$mb = (Get-Item $out).Length / 1MB
Write-Host ("DONE -> {0}  ({1:N1} MB)" -f $out, $mb) -ForegroundColor Green
Write-Host "Upload to EC2 with:" -ForegroundColor Yellow
Write-Host '  scp -i "<path_to_key.pem>" "' + $out + '" ubuntu@<PUBLIC-IP>:~'
