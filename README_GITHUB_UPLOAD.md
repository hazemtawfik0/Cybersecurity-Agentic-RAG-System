# GitHub upload pack

Target folder:

```text
C:\Users\Tawfik\Downloads\trae\cyber
```

Target repository:

```text
https://github.com/hazemtawfik0/Cybersecurity-Agentic-RAG-System.git
```

## Steps

1. Extract the three files from this ZIP into the `cyber` folder.
2. Confirm that `.gitignore` is beside `app.py`.
3. Double-click `check_github_files.bat`.
4. Double-click `upload_to_github.bat`.
5. Sign in to GitHub in the browser window if Git Credential Manager asks.

The script excludes `.venv`, `.hf_cache`, downloaded model files, Python caches,
logs, and secret files.

Do not delete `.venv`; it remains on the computer but is not committed.
