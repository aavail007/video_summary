@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 無法建立 Python 虛擬環境。請先安裝 Python 3.10 以上版本。
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if not exist ".env" copy ".env.example" ".env" >nul

echo.
echo 安裝完成。請執行 run.bat。
pause
