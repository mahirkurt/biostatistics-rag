@echo off
echo ====================================================
echo Biostatistics PDF Indexer (External Terminal)
echo ====================================================
echo.
echo This runs the indexer in a separate window to avoid 
echo VS Code terminal interrupts.
echo.
echo Press Ctrl+C in THIS window to stop.
echo.

:loop
echo [%date% %time%] Starting indexer...
python rag_system\indexers\run_indexer_pure_onnx.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Indexing completed!
    pause
    exit /b 0
)

echo.
echo [RESTART] Restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto loop
