# Architecture

## Offline lifecycle

Approved sources -> source registry -> local ingestion -> data quality -> cleaning/chunking -> fingerprints/versioning -> knowledge-base manifest -> local retrieval index. Evaluation, model registry and promotion helpers are available separately.

## Online request lifecycle

1. Request enters API.
2. Rule-based safety policy classifies intent and risk.
3. Matched individualized requests, emergencies and out-of-domain requests abstain or escalate.
4. Local TF-IDF cosine and lexical overlap retrieval return candidates.
5. Reranker selects final evidence.
6. Evidence gate checks sufficiency.
7. Extractive or optional base-model generator produces a response.
8. Lexical claim verifier estimates support.
9. Composite heuristic confidence and risk-specific threshold decide ANSWER/ABSTAIN.
10. Response, evidence, decision and confidence are logged to SQLite.
11. Feedback, monitoring and offline evaluation provide inputs for further development.

## Production boundaries

The repository separates policy, retrieval, generation, verification, confidence and presentation so components can be evaluated, versioned and replaced. The query rewriting, experiment tracking, PHI redaction, drift and refresh utilities are not fully integrated into the request path. The local demo is not a deployed, clinically validated or production-ready healthcare system.
