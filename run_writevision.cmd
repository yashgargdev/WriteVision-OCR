@echo off
setlocal enableextensions
pushd "%~dp0"

REM Detect Python
where py >nul 2>&1
if %errorlevel% EQU 0 (
  set "PYTHON=py -3"
) else (
  where python >nul 2>&1
  if %errorlevel% EQU 0 (
    set "PYTHON=python"
  ) else (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.10+ and rerun this script.
    goto :end
  )
)

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYTHON% -m venv ".venv"
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM Optional: override model (uncomment to switch)
REM set WRITEVISION_MODEL=microsoft\trocr-large-handwritten

REM Check dependencies by importing key packages
echo Checking dependencies...
python -c "import flask, PIL, transformers, torch, safetensors" >nul 2>&1
if %errorlevel% neq 0 (
  echo Dependencies missing. Installing...
  python -m pip install --upgrade pip
  pip install -r "requirements.txt"
  if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    goto :end
  )
) else (
  echo Dependencies already installed.
)

REM If fine-tuned model exists, use it by default
set "LOCAL_MODEL=models\writevision-trocr-finetuned"
if exist "%LOCAL_MODEL%\config.json" (
  set WRITEVISION_MODEL=%LOCAL_MODEL%
  echo Using local fine-tuned model: %WRITEVISION_MODEL%
) else (
  echo No local fine-tuned model found. Using base: %WRITEVISION_MODEL%
)

REM Handle training mode via label to avoid paren parsing issues
if /I "%~1"=="train" goto :train_mode

echo Starting WriteVision server...
python "app.py"
if %errorlevel% neq 0 (
  echo Server exited with an error.
)

goto :end

:train_mode
set EPOCHS=%~2
if "%EPOCHS%"=="" set EPOCHS=3
set BATCH=%~3
if "%BATCH%"=="" set BATCH=2
set SAVEDIR=%~4
if "%SAVEDIR%"=="" set SAVEDIR=%LOCAL_MODEL%

echo Starting training on data/train for %EPOCHS% epochs, batch %BATCH%...
python "train.py" --epochs %EPOCHS% --batch_size %BATCH% --save_dir "%SAVEDIR%"
if %errorlevel% neq 0 (
  echo Training failed. Aborting.
  goto :end
)
echo Training complete.
set WRITEVISION_MODEL=%SAVEDIR%

echo Starting WriteVision server...
python "app.py"
if %errorlevel% neq 0 (
  echo Server exited with an error.
)

goto :end

:end
popd
pause