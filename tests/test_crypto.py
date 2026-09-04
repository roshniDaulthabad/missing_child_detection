import os
import pytest
from app.security.crypto import CryptoManager
from app.security.hashing import IntegrityService

def test_aes_256_gcm_roundtrip(tmp_path):
    mgr = CryptoManager()
    sample_text = b"Confidential biometrics of missing child ID: MC-2026-001"
    
    # Encrypt
    encrypted = mgr.encrypt_bytes(sample_text)
    assert encrypted != sample_text
    assert len(encrypted) >= 12 + len(sample_text) + 16
    
    # Decrypt
    decrypted = mgr.decrypt_bytes(encrypted)
    assert decrypted == sample_text

def test_aes_tamper_detection():
    mgr = CryptoManager()
    sample_text = b"Sensitive face embedding payload"
    encrypted = bytearray(mgr.encrypt_bytes(sample_text))
    
    # Tamper with the ciphertext byte
    encrypted[-1] ^= 0xFF
    
    with pytest.raises(Exception):
        mgr.decrypt_bytes(bytes(encrypted))

def test_sha3_256_hashing(tmp_path):
    test_file = tmp_path / "sample.jpg"
    content = b"\xFF\xD8\xFF\xE0" + b"dummy image binary bytes"
    test_file.write_bytes(content)
    
    hash_val = IntegrityService.hash_file(str(test_file))
    assert len(hash_val) == 64  # SHA-3-256 produces 64 hex characters
    
    # Verify match
    assert IntegrityService.verify_file_integrity(str(test_file), hash_val) is True
    
    # Verify tampered mismatch
    test_file.write_bytes(content + b"tamper")
    assert IntegrityService.verify_file_integrity(str(test_file), hash_val) is False
