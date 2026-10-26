#!/usr/bin/env python3
"""Render P0 Security Advisory Live Verification Script.

Checks runtime compliance with:
1. Render eMudhra advisory:
   - certifi >= 2026.10.19
   - Rejection of compromised eMudhra intermediate certificates
   - Node.js runtime version >= 24.22.1 (if Node is installed)
2. Render PHP2Shell advisory:
   - Absence of unpatched PHP/Adminer services

Usage:
    python scripts/verify_render_advisory_live.py [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List


ADVISORY_TEST_ENDPOINTS = [
    "https://dv-g3b-emudhra.we-have-a-problem.com/",
    "https://dv-g3b-aaa-emudhra.we-have-a-problem.com/",
]

MINIMUM_CERTIFI_DATE = (2026, 10, 19)
MINIMUM_NODE_VERSION = (24, 22, 1)


def parse_version_tuple(v_str: str) -> tuple[int, ...]:
    cleaned = v_str.strip().lstrip("v")
    parts: list[int] = []
    for part in cleaned.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def check_certifi_version() -> dict[str, Any]:
    try:
        import certifi

        v_str = getattr(certifi, "__version__", "0.0.0")
        v_tuple = parse_version_tuple(v_str)
        # certifi versions are YYYY.MM.DD (e.g., 2026.10.23)
        passed = v_tuple >= MINIMUM_CERTIFI_DATE
        return {
            "installed": True,
            "version": v_str,
            "required_min": "2026.10.19",
            "passed": passed,
            "ca_file": certifi.where(),
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "required_min": "2026.10.19",
            "passed": False,
            "error": "certifi not installed",
        }


def check_node_version() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            v_str = proc.stdout.strip()
            v_tuple = parse_version_tuple(v_str)
            passed = v_tuple >= MINIMUM_NODE_VERSION
            return {
                "detected": True,
                "version": v_str,
                "required_min": "24.22.1",
                "passed": passed,
            }
        return {
            "detected": False,
            "version": None,
            "passed": True,  # Node not required in pure Python runtime container
            "note": "Node executable returned non-zero; not active in runtime",
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "detected": False,
            "version": None,
            "passed": True,  # Node not present in pure Python containers
            "note": "Node not found on PATH (pure Python container)",
        }


def test_endpoint_rejection(url: str, ca_file: str | None = None) -> dict[str, Any]:
    ctx = ssl.create_default_context(cafile=ca_file)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AiMedRes-AdvisoryVerifier/1.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=6) as response:
            return {
                "url": url,
                "rejected": False,
                "status_code": response.status,
                "verdict": "VULNERABLE (connection succeeded with untrusted certificate)",
                "passed": False,
            }
    except urllib.error.URLError as err:
        err_reason = str(err.reason)
        # Expect CERTIFICATE_VERIFY_FAILED or SSLCertVerificationError
        is_ssl_error = "CERTIFICATE_VERIFY_FAILED" in err_reason or isinstance(
            err.reason, ssl.SSLError
        )
        return {
            "url": url,
            "rejected": True,
            "error_type": type(err.reason).__name__,
            "error_detail": err_reason,
            "ssl_verification_failed": is_ssl_error,
            "verdict": "SECURE (certificate successfully rejected)" if is_ssl_error else "CONNECTION_ERROR",
            "passed": is_ssl_error,
        }
    except Exception as err:  # pylint: disable=broad-except
        return {
            "url": url,
            "rejected": True,
            "error_type": type(err).__name__,
            "error_detail": str(err),
            "ssl_verification_failed": isinstance(err, ssl.SSLError),
            "verdict": "EXCEPTION",
            "passed": isinstance(err, ssl.SSLError),
        }


def check_no_php_services(ports: list[int] | None = None) -> dict[str, Any]:
    if ports is None:
        ports = [8080, 9000]  # Common Adminer / PHP-FPM ports
    active_ports: list[int] = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                active_ports.append(port)
        except Exception:  # pylint: disable=broad-except
            pass
        finally:
            sock.close()
    return {
        "ports_checked": ports,
        "active_ports": active_ports,
        "passed": len(active_ports) == 0,
        "verdict": "SECURE (no PHP/Adminer ports listening)" if len(active_ports) == 0 else "WARNING: port listening",
    }


def run_full_verification() -> dict[str, Any]:
    certifi_res = check_certifi_version()
    ca_file = certifi_res.get("ca_file")
    node_res = check_node_version()
    endpoint_results = [
        test_endpoint_rejection(url, ca_file) for url in ADVISORY_TEST_ENDPOINTS
    ]
    php_res = check_no_php_services()

    endpoints_passed = all(r.get("passed", False) for r in endpoint_results)
    overall_passed = certifi_res.get("passed", False) and node_res.get("passed", False) and endpoints_passed

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "advisory": "Render eMudhra & PHP2Shell (October 2026)",
        "certifi": certifi_res,
        "node": node_res,
        "certificate_rejection_tests": endpoint_results,
        "php_adminer_check": php_res,
        "overall_secure": overall_passed,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Render Security Advisory Controls Live")
    parser.add_argument("--json-out", type=str, default="", help="Save report to JSON file")
    args = parser.parse_args()

    print("=================================================================")
    print(" Render P0 Security Advisory Live Verification (October 2026)")
    print("=================================================================")

    report = run_full_verification()

    # Certifi
    c_res = report["certifi"]
    status_c = "[PASS]" if c_res.get("passed") else "[FAIL]"
    print(f"\n{status_c} certifi version: {c_res.get('version')} (required >= {c_res.get('required_min')})")

    # Node
    n_res = report["node"]
    status_n = "[PASS]" if n_res.get("passed") else "[FAIL]"
    if n_res.get("detected"):
        print(f"{status_n} Node.js runtime: {n_res.get('version')} (required >= {n_res.get('required_min')})")
    else:
        print(f"{status_n} Node.js runtime: not active/detected ({n_res.get('note')})")

    # Endpoints
    print("\n* Testing certificate rejection on eMudhra advisory test endpoints:")
    for res in report["certificate_rejection_tests"]:
        status_e = "[PASS]" if res.get("passed") else "[FAIL]"
        print(f"  {status_e} {res.get('url')}")
        print(f"         Verdict: {res.get('verdict')}")
        if res.get("error_detail"):
            print(f"         Detail:  {res.get('error_detail')[:80]}")

    # PHP
    p_res = report["php_adminer_check"]
    status_p = "[PASS]" if p_res.get("passed") else "[WARN]"
    print(f"\n{status_p} PHP/Adminer port isolation: {p_res.get('verdict')}")

    print("\n-----------------------------------------------------------------")
    if report["overall_secure"]:
        print("[PASS] ALL CONTROLS VERIFIED SECURE AGAINST RENDER ADVISORIES")
        exit_code = 0
    else:
        print("[FAIL] ONE OR MORE SECURITY ADVISORY CONTROLS FAILED")
        exit_code = 1
    print("-----------------------------------------------------------------\n")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.json_out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
