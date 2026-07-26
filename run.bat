@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 第一次執行，開始安裝環境...
  call setup.bat
)

if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" app.py

