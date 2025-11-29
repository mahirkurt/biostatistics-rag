<# 
Auto-restart indexer script
Runs the indexer in a loop, automatically restarting after interrupts
#>

Write-Host "====================================="
Write-Host "Auto-Restart Biostatistics Indexer"
Write-Host "====================================="
Write-Host ""

$maxAttempts = 100
$attempt = 0
$pythonScript = "rag_system/indexers/run_indexer_pure_onnx.py"

while ($attempt -lt $maxAttempts) {
    $attempt++
    Write-Host "[Attempt $attempt] Starting indexer..."
    
    try {
        python $pythonScript
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "[SUCCESS] Indexing completed!"
            break
        }
        else {
            Write-Host ""
            Write-Host "[RESTART] Exit code: $exitCode - Restarting in 2 seconds..."
            Start-Sleep -Seconds 2
        }
    }
    catch {
        Write-Host ""
        Write-Host "[ERROR] $($_.Exception.Message)"
        Write-Host "[RESTART] Restarting in 2 seconds..."
        Start-Sleep -Seconds 2
    }
}

Write-Host ""
Write-Host "Auto-restart loop ended after $attempt attempts"
