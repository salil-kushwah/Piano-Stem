@echo off
title Piano Stem Studio (Native Desktop GUI)
echo Starting Piano Stem Studio Native Desktop App...
cd /d "%~dp0"
start "" ".\.venv\Scripts\python.exe" desktop_app.py
exit
