# Database Schema & Data Models

## 1. ORM Architecture
Hackwave uses Flask-SQLAlchemy with an abstraction layer:
- Local Development Default: SQLite (`data/hackwave.db`)
- Production Ready: PostgreSQL (configured via `DATABASE_URL=postgresql://user:pass@host:5432/hackwave`)

## 2. Core Tables
- `users`: User identity, password hashes, roles (`officer`, `investigator`, `admin`).
- `missing_children`: Case metadata, child details, parent contacts, status, and assigned officer.
- `child_embeddings`: 512-D float vectors serialized with reference type (`original`, `age_progressed_+2y`, etc.).
- `cameras`: Camera IDs, locations, encrypted RTSP endpoints, GPS coordinates.
- `tracks`: Multi-object tracker states, camera associations, and first/last seen timestamps.
- `track_observations`: Individual frame detections, bounding boxes, face quality scores.
- `cross_camera_matches`: Sighting links across cameras with transition times.
- `alerts`: Potential match alerts awaiting verification.
- `recovery_records`: Post-location recovery metadata and legal guardian verification.
- `welfare_records`: Medical checkups, psychological counseling referrals, shelter placement.
- `audit_logs`: Immutable audit trail with SHA-3 integrity hashes.
