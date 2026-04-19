@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

if exist "%ROOT%\.venv2\Scripts\activate.bat" (
  call "%ROOT%\.venv2\Scripts\activate.bat"
) else (
  echo [WARN] .venv2 activation script not found. Continuing with system Python.
)

cd /d "%ROOT%arch-2\implementation"
python -m committee_llm.prepare_2wiki %*

exit /b %ERRORLEVEL%
