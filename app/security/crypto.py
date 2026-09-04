import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from app.config import Config

class CryptoManager:
    """Handles AES-256-GCM authenticated encryption and decryption for sensitive biometric files."""

    def __init__(self, key_bytes: bytes = None):
        if key_bytes is None:
            # Decode key from config or environment
            b64_key = Config.AES_SECRET_KEY
            try:
                raw = base64.b64decode(b64_key)
                if len(raw) >= 32:
                    self.key = raw[:32]
                else:
                    self.key = raw.ljust(32, b"0")
            except Exception:
                self.key = b64_key.encode().ljust(32, b"0")[:32]
        else:
            self.key = key_bytes[:32]

        self.aesgcm = AESGCM(self.key)

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Encrypts data using AES-256-GCM. Returns nonce (12 bytes) + ciphertext + tag (16 bytes)."""
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        """Decrypts AES-256-GCM payload. Validates authentication tag to prevent tampering."""
        if len(encrypted_data) < 28: # 12 bytes nonce + 16 bytes auth tag minimum
            raise ValueError("Encrypted data payload is too short or corrupted.")
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)

    def encrypt_file(self, src_path: str, dest_path: str):
        """Encrypts a file and writes ciphertext to destination path."""
        with open(src_path, "rb") as f:
            data = f.read()
        enc_data = self.encrypt_bytes(data)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(enc_data)

    def decrypt_file(self, src_path: str) -> bytes:
        """Reads an encrypted file and returns the decrypted plaintext bytes."""
        with open(src_path, "rb") as f:
            enc_data = f.read()
        return self.decrypt_bytes(enc_data)

# Singleton instance
crypto_service = CryptoManager()
