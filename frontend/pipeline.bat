@echo off
chcp 65001 >nul
echo ========================================
echo   Competency Gap Analyzer Pipeline
echo ========================================
echo.

echo [1/4] Collecting IT vacancies from hh.ru...
python main.py --it-sector --regions %1 --excel
if errorlevel 1 goto error
echo [1/4] ✓ Vacancies collected successfully
echo.

echo [2/4] Training vacancy clusters...
python scripts/train_clusters.py --level all
if errorlevel 1 goto error
echo [2/4] ✓ Clusters trained successfully
echo.

echo [3/4] Training LTR ranking model...
python main.py --train-model
if errorlevel 1 goto error
echo [3/4] ✓ Model trained successfully
echo.

echo [4/4] Running GAP analysis...
python main.py --skip-collection --run-gap-analysis
if errorlevel 1 goto error
echo [4/4] ✓ GAP analysis completed
echo.

echo ========================================
echo   ✓ All steps completed successfully!
echo ========================================
goto end

:error
echo.
echo ========================================
echo   ✗ Error occurred! Pipeline stopped.
echo ========================================
exit /b 1

:end
exit /b 0
