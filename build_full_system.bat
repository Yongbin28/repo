@echo off
echo ==============================================
echo Building Wafer Quality Gate Desktop App
echo ==============================================

echo Step 0: Cleaning up existing processes and old builds...
taskkill /F /IM "T&P_Wafer_Quality_Gate.exe" /T 2>nul
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq T* Wafer Quality Gate*" /T 2>nul
taskkill /F /IM "msedge.exe" /FI "WINDOWTITLE eq Wafer Quality Gate*" /T 2>nul
rem Kill any stray Streamlit or WebView processes
taskkill /F /IM streamlit.exe /T 2>nul
taskkill /F /IM "T&P Wafer Quality Gate.exe" /T 2>nul
taskkill /F /IM "Wafer_Quality_Gate.exe" /T 2>nul

if exist desktop_app_log.txt del /F /Q desktop_app_log.txt
if exist build rd /s /q build
if exist dist (
    echo Cleaning dist folder...
    rd /s /q dist
)
if exist dist move dist "dist_old_%RANDOM%" 2>nul

echo Step 1/4: Copying local ms-playwright directory...
xcopy /E /I /H /Y "%LOCALAPPDATA%\ms-playwright" "ms-playwright"

echo Step 2/4: Creating virtual environment (fyp_env)...
python -m venv fyp_env
call fyp_env\Scripts\activate.bat

echo Step 3/4: Installing all requirements and packaging tools...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller pywebview cx_Freeze playwright

pyinstaller -y desktop_app.spec

echo ==============================================
echo Build Completed Successfully!
echo ==============================================
