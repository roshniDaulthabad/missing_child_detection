# Security & Cryptography Architecture

## 1. Biometric Data Protection (AES-256-GCM)
Child photographs and face crops contain sensitive biometric data. All stored images are encrypted at rest using AES-256 in Galois/Counter Mode (GCM):
- 12-byte random initialization vector (IV / nonce)
- Authenticated payload with 16-byte authentication tag
- Tampering with encrypted files immediately raises cryptographic verification exceptions

## 2. Integrity Verification (SHA-3-256)
Every uploaded image and significant investigation event is digested with SHA-3-256:
- Computed upon initial case enrollment
- Verified prior to decryption and display
- Audit trail logs are signed with cryptographic integrity hashes

## 3. Threat Mitigation
- No raw RTSP credentials exposed to frontend
- Role-based authorization on all case update endpoints
- Zero model retraining on case enrollment prevents model poisoning attacks
