@echo off
REM Helper script to run Python commands with venv activated
REM This bypasses PowerShell execution policy issues

call venv\Scripts\activate.bat
python %*

