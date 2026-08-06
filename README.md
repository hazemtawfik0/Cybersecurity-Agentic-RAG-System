# Cybersecurity Agentic RAG — Windows App v1.7

## Installation folder

Extract these files directly into:

```text
C:\Users\Tawfik\Downloads\trae\cyber
```

The app automatically finds the vector-store files beside `app.py`.

## Start

For the existing installation:

```text
run_app.bat
```

For a new installation:

```text
setup_and_run.bat
```

The browser opens at:

```text
http://127.0.0.1:7860
```

## Fixed in v1.2

- Removes the Windows trailing-quote path bug
- Automatically discovers the vector-store folder
- Adds an Advanced Setup path field and Reload button
- Adds a polished responsive dashboard
- Adds quick questions and a dedicated evidence tab
- Uses Fast Grounded Evidence by default for a quicker first test
- Remains compatible with Gradio 6

## Required vector-store files

```text
bm25_tokenized_corpus.json
chunk_embeddings.npy
chunks_metadata.jsonl
retrieval_config.json
```

## AI mode

The first AI-synthesis question downloads the selected Qwen model.  
Fast Grounded Evidence does not load Qwen and is the quickest mode.


## Answer-quality changes in v1.3

- Replaces the old one-sentence-per-chunk output
- Expands retrieval queries according to the question intent
- Ranks complete sentences across multiple candidate chunks
- Filters broken PDF fragments, headings, URLs, and duplicates
- Returns one direct sentence for simple list questions
- Separates preparation and recovery for ransomware planning questions
- Prioritizes isolation and containment for immediate-response questions
- Displays only sources that are actually cited in the answer


## Answer-quality changes in v1.4

- Adds a dedicated NIST SP 800-30 risk-assessment intent
- Prioritizes the four stages: Prepare, Conduct, Communicate, and Maintain
- Strongly penalizes unrelated CSF, incident-response, and collaboration text
- Removes extraction prefixes such as `N1:`
- Gives additional weight to practical ransomware actions over generic framework descriptions


## Answer-quality changes in v1.5

- Returns the six CSF Functions as a direct exact list
- Returns all four NIST SP 800-30 stages with correctly matched tasks
- Returns the seven NIST SP 800-207 zero-trust tenets
- Injects the required local source chunks before approximate hybrid results
- Allows more chunks from one document for multi-page standards
- Removes extraction artifacts such as `strategy-and` and footnote suffixes
- Removes generic framework prose from ransomware preparation/recovery answers
- Fixes the AI-mode fallback to use Smart Grounded Answer


## UI update in v1.6

- Wider desktop layout that uses much more of the page width
- Larger chat area with better viewport fit
- Sticky right sidebar for controls
- Quick questions moved into a compact accordion
- Tighter hero section and denser spacing for presentation use
- Improved overall fit on 1080p laptop screens


## UI update in v1.7

- Restores suggested questions as visible buttons
- Keeps them compact so the page still fits well
- Clicking a suggestion places the full question in the input box
