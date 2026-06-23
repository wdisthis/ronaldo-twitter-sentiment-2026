@echo off
title Ronaldo Twitter Sentiment App Runner
echo =======================================================
echo     CRISTIANO RONALDO TWITTER SENTIMENT APP RUNNER
echo =======================================================
echo.

:: Check for virtual environment and activate it
if exist "%~dp0.env\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment in .env...
    call "%~dp0.env\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment '.env' not found in this folder.
    echo [INFO] Attempting to run Streamlit using global python environment...
)

echo.
echo [INFO] Launching Streamlit application app.py...
streamlit run "%~dp0app.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit. Please verify that python and streamlit are installed.
    pause
)
