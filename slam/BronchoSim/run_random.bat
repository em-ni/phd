@echo off
REM Usage: run_random.bat N
REM Runs the simulation N times with random selection of 3 centerlines in autopilot mode

if "%~1"=="" (
  echo Usage: %0 N
  exit /b 1
)

set N=%1
for /L %%i in (1,1,%N%) do (
  echo Run %%i/%N%:
  python main.py -view fp -random 3 -autopilot -record True
  echo ---
)
