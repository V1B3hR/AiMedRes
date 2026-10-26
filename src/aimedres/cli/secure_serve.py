"""Supported command-line entry point for the maintained research API."""

from __future__ import annotations

import argparse
import os
import ssl
from typing import Optional, Sequence

from aimedres.service import SecureAPIServer


def _production_mode() -> bool:
    return os.environ.get("AIMEDRES_ENV", "").lower() == "production" or os.environ.get(
        "STRICT_SECURITY", "false"
    ).lower() in {"1", "true"}


def _public_bind(host: str) -> bool:
    return host not in {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the maintained AiMedRes research API (not for clinical use)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable the Flask debugger locally")
    parser.add_argument("--ssl-cert", help="PEM certificate file for direct TLS")
    parser.add_argument("--ssl-key", help="PEM private-key file for direct TLS")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    production = _production_mode()

    if args.debug and production:
        parser.error("--debug is prohibited when AIMEDRES_ENV=production or STRICT_SECURITY=true")
    if _public_bind(args.host) and not production:
        parser.error(
            "A non-loopback bind requires AIMEDRES_ENV=production (or STRICT_SECURITY=true) "
            "and explicitly configured FLASK_SECRET_KEY and JWT_SECRET."
        )
    if bool(args.ssl_cert) != bool(args.ssl_key):
        parser.error("--ssl-cert and --ssl-key must be supplied together")

    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)

    server = SecureAPIServer()
    server.run(host=args.host, port=args.port, debug=args.debug, ssl_context=ssl_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
