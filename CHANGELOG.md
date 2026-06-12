## v1.0.0
2026-06-13

### Changed

- Replace QdrantClient with AsyncQdrantClient
- Migrate collection initialization into FastAPI lifespan
- Remove Import Side Effect in database.py
- Refactor retrieval pipeline to async

### Fixed

- Fix pseudo-async retrieval chain caused by sync Qdrant client
