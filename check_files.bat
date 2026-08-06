@echo off
setlocal
cd /d "%~dp0"

echo Checking required vector-store files...
echo.

set missing=0

for %%F in (
    bm25_tokenized_corpus.json
    chunk_embeddings.npy
    chunks_metadata.jsonl
    retrieval_config.json
) do (
    if exist "%%F" (
        echo [OK] %%F
    ) else (
        echo [MISSING] %%F
        set missing=1
    )
)

echo.
if "%missing%"=="1" (
    echo Some required files are missing.
    echo Extract the Gradio app files directly into the same folder
    echo as your vector-store files.
) else (
    echo All required vector-store files are present.
)

pause
