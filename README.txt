Cybersecurity Agentic RAG Artifacts — Version 4
================================================

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Generation model used in the notebook:
Qwen/Qwen2.5-1.5B-Instruct

Chunks:
339

Retrieval evaluation questions:
40

Main improvements:
- robust subquery parsing
- hybrid retrieval
- section-fragment filtering
- inline citation enforcement
- citation repair
- structured JSON answer generation
- programmatic inline citation rendering
- strict source-label validation
- semantic support filtering
- one generation call per answer
- balanced multi-query evidence selection
- expanded retrieval evaluation

Files:
- chunk_embeddings.npy
- cybersecurity_chunks.faiss
- chunks_metadata.jsonl
- bm25_tokenized_corpus.json
- retrieval_config.json
- retrieval_evaluation.csv
- retrieval_evaluation_by_category.csv
- example_agent_output.json

The generation model is not included. The local Gradio application will
download or load the selected generation model separately.