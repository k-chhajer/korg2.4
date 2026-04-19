@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

if exist "%ROOT%\.venv2\Scripts\activate.bat" (
  call "%ROOT%\.venv2\Scripts\activate.bat"
) else (
  echo [WARN] .venv2 activation script not found. Continuing with system Python.
)

set "ENV_FILE_KEY="
if exist "%ROOT%\.env" (
  for /f "usebackq tokens=* delims=" %%L in ("%ROOT%\.env") do (
    set "line=%%L"
    if not "!line!"=="" if /I not "!line:~0,1!"=="#" (
      if /I "!line:~0,7!"=="export " set "line=!line:~7!"
      for /f "tokens=1,* delims==" %%A in ("!line!") do (
        if /I "%%A"=="OPENROUTER_API_KEY" set "ENV_FILE_KEY=%%B"
      )
    )
  )
)

if defined ENV_FILE_KEY set "OPENROUTER_API_KEY=%ENV_FILE_KEY%"

if defined OPENROUTER_API_KEY (
  set "OPENROUTER_API_KEY=%OPENROUTER_API_KEY:"=%"
)

if not defined OPENROUTER_API_KEY (
  echo [ERROR] OPENROUTER_API_KEY is not set.
  echo Set it in environment or in %ROOT%\.env, then run again.
  exit /b 1
)

if /I "%OPENROUTER_API_KEY%"=="YOUR_OPENROUTER_KEY" (
  echo [ERROR] OPENROUTER_API_KEY is a placeholder value.
  exit /b 1
)

if /I "%OPENROUTER_API_KEY%"=="YOUR_OPENROUTER_API_KEY" (
  echo [ERROR] OPENROUTER_API_KEY is a placeholder value.
  exit /b 1
)

set "MASKED_KEY=%OPENROUTER_API_KEY:~0,12%..."
echo [INFO] OPENROUTER_API_KEY loaded: !MASKED_KEY!

python -c "import sentence_transformers" >nul 2>&1
if errorlevel 1 (
  echo [WARN] sentence-transformers is not installed. Semantic embeddings will fall back to zeros.
  echo [WARN] Install with: pip install -r "%ROOT%arch-2\implementation\requirements.txt"
)

set "CONFIG=%ROOT%arch-2\implementation\configs\qwen3_8b_openrouter_arch2_controller.json"
set "TASKS=%ROOT%arch-2\evals\data\benchmarks\2wikimultihop_train.jsonl"
set "OUTDIR=%ROOT%arch-2\runs\arch2_controller_2wiki_cpu"

if not exist "%TASKS%" (
  echo [ERROR] Converted 2Wiki training file not found:
  echo   %TASKS%
  echo Run prepare_arch2_2wiki.bat first.
  exit /b 1
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

cd /d "%ROOT%arch-2\implementation"
python -m committee_llm.train_controller ^
  --config "%CONFIG%" ^
  --tasks "%TASKS%" ^
  --outdir "%OUTDIR%" ^
  --limit 5000 ^
  --episodes 120 ^
  --eval-every 10 ^
  --save-every 5 ^
  --eval-task-count 24 ^
  --grpo-group-size 4 ^
  --phase-loss-weight 0.1 ^
  --entropy-coef 0.01 ^
  --cheap-model anthropic/claude-haiku-4-5-20251001 ^
  --semantic-model all-MiniLM-L6-v2 ^
  --budget-tokens 16000 ^
  --max-decisions 6 ^
  --max-restarts 1 ^
  --per-role-call-cap 2 ^
  --device cpu %*

exit /b %ERRORLEVEL%
