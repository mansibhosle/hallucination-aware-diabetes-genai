# Project Roadmap

## Phase 1 — Local lifecycle (included)
- CPU-friendly hybrid retrieval
- deterministic safety policies
- extractive grounded fallback
- confidence/abstention
- FastAPI, Streamlit, SQLite
- evaluation + tests

## Phase 2 — Capstone models (integration code added; runtime validation pending)
- load existing Phi-3 SFT adapter with 4-bit inference
- load Sentence-BERT all-MiniLM-L6-v2 and validated private FAISS index
- run GPU smoke tests, preserve the historical evaluation and evaluate changed system on independent questions

## Phase 3 — Reliability upgrades
- BM25 + FAISS hybrid retrieval
- cross-encoder reranking and score normalization
- clinical NLI claim verification
- learned/calibrated confidence model
- broader safety intent classifier
- external benchmark + clinician review

## Phase 4 — MLOps/LLMOps
- MLflow/W&B experiment tracking
- PostgreSQL + object storage
- Qdrant/pgvector/OpenSearch
- scheduled knowledge refresh
- drift alerts
- model promotion/rollback
- CI/CD gates based on evaluation thresholds

## Phase 5 — Cloud deployment
- Docker images
- FastAPI service behind authenticated gateway
- Streamlit/React frontend
- managed secrets/KMS
- privacy-preserving audit retention policies
- dashboards/alerts

Clinical use requires independent validation, privacy/security review and applicable regulatory assessment.
