"""
API Routes for Quantum-Safe Cryptography (P3-2)

Provides endpoints for:
- Quantum-safe key exchange
- Hybrid encryption (classical + post-quantum)
- Key rotation management
- KMS integration
- Migration path monitoring
"""

import base64
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger("aimedres.api.quantum_routes")

# Create blueprint
quantum_bp = Blueprint("quantum", __name__, url_prefix="/api/v1/quantum")

# Import quantum crypto components
try:
    import os
    import sys

    security_path = os.path.join(os.path.dirname(__file__), "../../../security")
    if security_path not in sys.path:
        sys.path.insert(0, security_path)

    from quantum_crypto import QuantumSafeCryptography
    from quantum_prod_keys import QuantumProductionKeyManager

    QUANTUM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantum crypto module not available: {e}")
    QUANTUM_AVAILABLE = False

from ..security.auth import require_admin, require_auth

# Initialize quantum crypto
quantum_crypto = None
key_manager = None

if QUANTUM_AVAILABLE:
    try:
        quantum_config = {
            "quantum_safe_enabled": True,
            "quantum_algorithm": "kyber768",
            "hybrid_mode": True,
            "classical_algorithm": "AES-256-GCM",
            "performance_monitoring": True,
        }
        quantum_crypto = QuantumSafeCryptography(quantum_config)

        kms_config = {
            "provider": "aws_kms",
            "region": "us-east-1",
            "rotation_days": 90,
            "quantum_safe_enabled": True,
        }
        key_manager = QuantumProductionKeyManager(kms_config)
    except Exception as e:
        logger.error(f"Failed to initialize quantum crypto: {e}")


# ==================== Key Exchange ====================


@quantum_bp.route("/key-exchange/init", methods=["POST"])
@require_auth
def init_key_exchange():
    """
    Initialize quantum-safe key exchange.

    Request body:
    {
        "client_id": "client-123",
        "algorithm": "kyber768"  // optional
    }

    Returns:
        Public key and session parameters
    """
    try:
        if not QUANTUM_AVAILABLE or not quantum_crypto:
            return jsonify({"error": "Quantum crypto not available"}), 503

        data = request.get_json()
        client_id = data.get("client_id")
        algorithm = data.get("algorithm", "kyber768")

        if not client_id:
            return jsonify({"error": "client_id is required"}), 400

        # Generate key pair for exchange
        result = quantum_crypto.generate_keypair(algorithm=algorithm)

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "session_id": result.get("session_id"),
                        "public_key": base64.b64encode(result.get("public_key", b"")).decode(
                            "utf-8"
                        ),
                        "algorithm": algorithm,
                        "expires_at": result.get("expires_at"),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error initializing key exchange: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/key-exchange/complete", methods=["POST"])
@require_auth
def complete_key_exchange():
    """
    Complete quantum-safe key exchange.

    Request body:
    {
        "session_id": "session-123",
        "client_public_key": "base64_encoded_key"
    }

    Returns:
        Shared secret (encrypted) and confirmation
    """
    try:
        if not QUANTUM_AVAILABLE or not quantum_crypto:
            return jsonify({"error": "Quantum crypto not available"}), 503

        data = request.get_json()
        session_id = data.get("session_id")
        client_public_key_b64 = data.get("client_public_key")

        if not session_id or not client_public_key_b64:
            return jsonify({"error": "session_id and client_public_key are required"}), 400

        # Decode public key
        client_public_key = base64.b64decode(client_public_key_b64)

        # Complete key exchange
        result = quantum_crypto.complete_key_exchange(
            session_id=session_id, client_public_key=client_public_key
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "session_id": session_id,
                        "shared_secret_encrypted": base64.b64encode(
                            result.get("shared_secret_encrypted", b"")
                        ).decode("utf-8"),
                        "confirmation_hash": result.get("confirmation_hash"),
                        "algorithm_used": result.get("algorithm"),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error completing key exchange: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Hybrid Encryption ====================


@quantum_bp.route("/encrypt", methods=["POST"])
@require_auth
def hybrid_encrypt():
    """
    Encrypt data using hybrid encryption (classical + post-quantum).

    Request body:
    {
        "data": "base64_encoded_plaintext",
        "recipient_public_key": "base64_encoded_key",
        "algorithm": "hybrid_kyber_aes"  // optional
    }

    Returns:
        Encrypted data and metadata
    """
    try:
        if not QUANTUM_AVAILABLE or not quantum_crypto:
            return jsonify({"error": "Quantum crypto not available"}), 503

        data = request.get_json()
        plaintext_b64 = data.get("data")
        recipient_key_b64 = data.get("recipient_public_key")
        algorithm = data.get("algorithm", "hybrid_kyber_aes")

        if not plaintext_b64:
            return jsonify({"error": "data is required"}), 400

        # Decode plaintext
        plaintext = base64.b64decode(plaintext_b64)
        recipient_key = base64.b64decode(recipient_key_b64) if recipient_key_b64 else None

        # Encrypt
        result = quantum_crypto.hybrid_encrypt(
            plaintext=plaintext, recipient_public_key=recipient_key, algorithm=algorithm
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "ciphertext": base64.b64encode(result.get("ciphertext", b"")).decode(
                            "utf-8"
                        ),
                        "algorithm": algorithm,
                        "key_id": result.get("key_id"),
                        "nonce": base64.b64encode(result.get("nonce", b"")).decode("utf-8"),
                        "tag": base64.b64encode(result.get("tag", b"")).decode("utf-8"),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error encrypting data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/decrypt", methods=["POST"])
@require_auth
def hybrid_decrypt():
    """
    Decrypt data using hybrid decryption.

    Request body:
    {
        "ciphertext": "base64_encoded_ciphertext",
        "key_id": "key-123",
        "nonce": "base64_encoded_nonce",
        "tag": "base64_encoded_tag",
        "algorithm": "hybrid_kyber_aes"
    }

    Returns:
        Decrypted plaintext
    """
    try:
        if not QUANTUM_AVAILABLE or not quantum_crypto:
            return jsonify({"error": "Quantum crypto not available"}), 503

        data = request.get_json()
        ciphertext_b64 = data.get("ciphertext")
        key_id = data.get("key_id")
        nonce_b64 = data.get("nonce")
        tag_b64 = data.get("tag")
        algorithm = data.get("algorithm", "hybrid_kyber_aes")

        if not ciphertext_b64:
            return jsonify({"error": "ciphertext is required"}), 400

        # Decode ciphertext
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64) if nonce_b64 else None
        tag = base64.b64decode(tag_b64) if tag_b64 else None

        # Decrypt
        result = quantum_crypto.hybrid_decrypt(
            ciphertext=ciphertext, key_id=key_id, nonce=nonce, tag=tag, algorithm=algorithm
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "plaintext": base64.b64encode(result.get("plaintext", b"")).decode("utf-8"),
                        "verified": result.get("verified", False),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error decrypting data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Key Management ====================


@quantum_bp.route("/keys", methods=["POST"])
@require_admin
def create_key():
    """
    Create a new quantum-safe key.

    Request body:
    {
        "key_type": "encryption",  // encryption, signing
        "algorithm": "kyber768",
        "purpose": "data_encryption",
        "rotation_policy": "90_days"
    }

    Returns:
        Key ID and metadata
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        data = request.get_json()
        key_type = data.get("key_type", "encryption")
        algorithm = data.get("algorithm", "kyber768")
        purpose = data.get("purpose", "data_encryption")
        rotation_policy = data.get("rotation_policy", "90_days")

        # Create key
        result = key_manager.create_key(
            key_type=key_type, algorithm=algorithm, purpose=purpose, rotation_policy=rotation_policy
        )

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "key_id": result.get("key_id"),
                        "algorithm": algorithm,
                        "created_at": result.get("created_at"),
                        "expires_at": result.get("expires_at"),
                        "rotation_policy": rotation_policy,
                    },
                }
            ),
            201,
        )

    except Exception as e:
        logger.error(f"Error creating key: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/keys/<key_id>/rotate", methods=["POST"])
@require_admin
def rotate_key(key_id: str):
    """
    Rotate a quantum-safe key.

    Returns:
        New key ID and rotation confirmation
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        # Rotate key
        result = key_manager.rotate_key(key_id=key_id)

        if not result:
            return jsonify({"error": "Key not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "data": {
                        "old_key_id": key_id,
                        "new_key_id": result.get("new_key_id"),
                        "rotated_at": result.get("rotated_at"),
                        "migration_status": result.get("migration_status"),
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error rotating key: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/keys", methods=["GET"])
@require_admin
def list_keys():
    """
    List all quantum-safe keys.

    Query params:
        status: Filter by status (active, rotated, revoked)

    Returns:
        List of keys
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        status_filter = request.args.get("status")

        keys = key_manager.list_keys(status=status_filter)

        return jsonify({"success": True, "data": {"keys": keys, "total_count": len(keys)}}), 200

    except Exception as e:
        logger.error(f"Error listing keys: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/keys/<key_id>", methods=["GET"])
@require_admin
def get_key_metadata(key_id: str):
    """
    Get key metadata.

    Returns:
        Key metadata (excluding private key material)
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        metadata = key_manager.get_key_metadata(key_id=key_id)

        if not metadata:
            return jsonify({"error": "Key not found"}), 404

        return jsonify({"success": True, "data": metadata}), 200

    except Exception as e:
        logger.error(f"Error getting key metadata: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Performance Monitoring ====================


@quantum_bp.route("/performance", methods=["GET"])
@require_auth
def get_performance_metrics():
    """
    Get quantum crypto performance metrics.

    Returns:
        Performance statistics and overhead analysis
    """
    try:
        if not QUANTUM_AVAILABLE or not quantum_crypto:
            return jsonify({"error": "Quantum crypto not available"}), 503

        metrics = quantum_crypto.get_performance_metrics()

        return jsonify({"success": True, "data": metrics}), 200

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Migration Path ====================


@quantum_bp.route("/migration/status", methods=["GET"])
@require_admin
def get_migration_status():
    """
    Get migration status from classical to quantum-safe crypto.

    Returns:
        Migration progress and statistics
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        status = key_manager.get_migration_status()

        return jsonify({"success": True, "data": status}), 200

    except Exception as e:
        logger.error(f"Error getting migration status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@quantum_bp.route("/migration/plan", methods=["POST"])
@require_admin
def create_migration_plan():
    """
    Create a migration plan to quantum-safe crypto.

    Request body:
    {
        "target_date": "2025-06-01",
        "phased_rollout": true,
        "backup_classical": true
    }

    Returns:
        Migration plan with timeline
    """
    try:
        if not QUANTUM_AVAILABLE or not key_manager:
            return jsonify({"error": "Key manager not available"}), 503

        data = request.get_json()
        target_date = data.get("target_date")
        phased_rollout = data.get("phased_rollout", True)
        backup_classical = data.get("backup_classical", True)

        plan = key_manager.create_migration_plan(
            target_date=target_date,
            phased_rollout=phased_rollout,
            backup_classical=backup_classical,
        )

        return jsonify({"success": True, "data": plan}), 201

    except Exception as e:
        logger.error(f"Error creating migration plan: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== Health Check ====================


@quantum_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for quantum crypto service."""
    return (
        jsonify(
            {
                "status": "healthy" if QUANTUM_AVAILABLE and quantum_crypto else "degraded",
                "service": "quantum_crypto",
                "timestamp": datetime.now().isoformat(),
                "quantum_crypto_available": QUANTUM_AVAILABLE,
                "algorithms_supported": (
                    ["kyber512", "kyber768", "kyber1024", "dilithium2", "dilithium3"]
                    if QUANTUM_AVAILABLE
                    else []
                ),
            }
        ),
        200,
    )
