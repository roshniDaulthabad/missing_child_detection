import hashlib
import os

class IntegrityService:
    """Provides SHA-3-256 hashing for file integrity and cryptographic tamper detection."""

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Computes SHA-3-256 hexadecimal digest of byte data."""
        return hashlib.sha3_256(data).hexdigest()

    @staticmethod
    def hash_file(filepath: str) -> str:
        """Computes SHA-3-256 hexadecimal digest of a file safely in chunks."""
        hasher = hashlib.sha3_256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def verify_file_integrity(filepath: str, expected_hash: str) -> bool:
        """Verifies if the file matches the expected SHA-3-256 hash."""
        if not os.path.exists(filepath):
            return False
        computed = IntegrityService.hash_file(filepath)
        return computed.lower() == expected_hash.lower()

integrity_service = IntegrityService()
