@echo off
title Piano Stem Studio
echo ===================================================
echo           PIANO STEM STUDIO - LAUNCHER
echo ===================================================
echo [1/2] Starting local AI pipeline server (FastAPI)...
cd /d "%~dp0"
start "Piano Stem Studio Server" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
echo [2/2] Opening Studio in your default browser...
start http://127.0.0.1:8000
echo ===================================================
echo Studio is running at: http://127.0.0.1:8000
echo Drop any MP3 to isolate piano, transcribe notes,
echo and watch the Rousseau falling notes visualizer!
echo ===================================================
pause
