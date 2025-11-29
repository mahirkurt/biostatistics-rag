@echo off
echo ================================================================================
echo BIOSTATISTICS RAG - BGE-M3 INDEXER
echo ================================================================================
echo.
echo Bu script BGE-M3 ile PDF indeksleme yapar.
echo VS Code terminal'de calismiyor - bu CMD penceresinde calistirin.
echo.

cd /d D:\Repositories\Biostatistics\rag_system\indexers

:loop
echo.
echo [%date% %time%] Indexer baslatiliyor...
python run_indexer_bge_m3.py

if errorlevel 1 (
    echo.
    echo [HATA] Indexer kesildi. 5 saniye sonra yeniden baslatilacak...
    timeout /t 5 /nobreak
    goto loop
)

echo.
echo ================================================================================
echo INDEKSLEME TAMAMLANDI!
echo ================================================================================
pause
