# Secure Data Transfer Methods Guide

## Overview

This guide outlines secure methods for transferring clinical data to and from the AiMedRes platform, ensuring HIPAA compliance and data protection during transit.

## Supported Transfer Methods

1. **SFTP (SSH File Transfer Protocol)** - Recommended for batch file transfers
2. **VPN (Virtual Private Network)** - For secure network-level access
3. **Secure REST APIs** - For real-time data integration

## Method 1: SFTP Configuration

### Prerequisites

- OpenSSH server (version 7.4+)
- SSH key pair for authentication
- Firewall rules configured for SFTP

### Server Setup

#### 1. Install and Configure SFTP Server

```bash
# Install OpenSSH server (if not already installed)
sudo apt-get update
sudo apt-get install openssh-server

# Create dedicated SFTP user for AiMedRes
sudo useradd -m -d /home/aimedres-sftp -s /bin/bash aimedres-sftp

# Create secure directory structure
sudo mkdir -p /home/aimedres-sftp/{upload,download,processed,quarantine}
sudo chown -R aimedres-sftp:aimedres-sftp /home/aimedres-sftp
sudo chmod 700 /home/aimedres-sftp
```

#### 2. Configure SSH for SFTP-only Access

Edit `/etc/ssh/sshd_config`:

```bash
# Add SFTP configuration
Match User aimedres-sftp
    ForceCommand internal-sftp
    PasswordAuthentication no
    PubkeyAuthentication yes
    ChrootDirectory /home/aimedres-sftp
    PermitTunnel no
    AllowAgentForwarding no
    AllowTcpForwarding no
    X11Forwarding no
```

Restart SSH service:

```bash
sudo systemctl restart sshd
```

#### 3. Generate and Configure SSH Keys

```bash
# Generate SSH key pair (on client machine)
ssh-keygen -t ed25519 -C "aimedres-sftp" -f ~/.ssh/aimedres_sftp_key

# Copy public key to server
ssh-copy-id -i ~/.ssh/aimedres_sftp_key.pub aimedres-sftp@your-server-ip

# Or manually add to authorized_keys
# On server:
sudo mkdir -p /home/aimedres-sftp/.ssh
sudo nano /home/aimedres-sftp/.ssh/authorized_keys
# Paste the public key content
sudo chmod 700 /home/aimedres-sftp/.ssh
sudo chmod 600 /home/aimedres-sftp/.ssh/authorized_keys
sudo chown -R aimedres-sftp:aimedres-sftp /home/aimedres-sftp/.ssh
```

### Client Configuration

#### Python SFTP Client Example

```python
import paramiko
import os
from pathlib import Path

class SecureSFTPClient:
    """Secure SFTP client for clinical data transfer"""
    
    def __init__(self, hostname, port=22, username='aimedres-sftp', 
                 key_path='~/.ssh/aimedres_sftp_key'):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.key_path = os.path.expanduser(key_path)
        self.client = None
        self.sftp = None
    
    def connect(self):
        """Establish secure SFTP connection"""
        try:
            # Load private key
            private_key = paramiko.Ed25519Key.from_private_key_file(self.key_path)
            
            # Create SSH client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
            
            # Load known hosts
            self.client.load_system_host_keys()
            
            # Connect with key authentication
            self.client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                pkey=private_key,
                timeout=30
            )
            
            # Open SFTP session
            self.sftp = self.client.open_sftp()
            print(f"✅ Connected to SFTP server: {self.hostname}")
            
        except Exception as e:
            print(f"❌ SFTP connection failed: {e}")
            raise
    
    def upload_file(self, local_path, remote_path):
        """Upload file with verification"""
        try:
            # Upload file
            self.sftp.put(local_path, remote_path)
            
            # Verify upload
            local_stat = os.stat(local_path)
            remote_stat = self.sftp.stat(remote_path)
            
            if local_stat.st_size == remote_stat.st_size:
                print(f"✅ Uploaded: {local_path} -> {remote_path}")
                return True
            else:
                print(f"❌ Upload verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def download_file(self, remote_path, local_path):
        """Download file with verification"""
        try:
            # Download file
            self.sftp.get(remote_path, local_path)
            
            # Verify download
            local_stat = os.stat(local_path)
            remote_stat = self.sftp.stat(remote_path)
            
            if local_stat.st_size == remote_stat.st_size:
                print(f"✅ Downloaded: {remote_path} -> {local_path}")
                return True
            else:
                print(f"❌ Download verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def list_files(self, remote_dir='/upload'):
        """List files in remote directory"""
        try:
            files = self.sftp.listdir(remote_dir)
            return files
        except Exception as e:
            print(f"❌ List files failed: {e}")
            return []
    
    def disconnect(self):
        """Close SFTP connection"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print("✅ SFTP connection closed")

# Usage example
if __name__ == '__main__':
    sftp = SecureSFTPClient(
        hostname='aimedres-server.hospital.org',
        username='aimedres-sftp',
        key_path='~/.ssh/aimedres_sftp_key'
    )
    
    try:
        sftp.connect()
        
        # Upload de-identified data
        sftp.upload_file('patient_data_deidentified.csv', '/upload/patient_data.csv')
        
        # Download results
        sftp.download_file('/download/results.json', 'results.json')
        
    finally:
        sftp.disconnect()
```

### SFTP Security Checklist

- [ ] SSH keys generated with strong algorithm (Ed25519 or RSA 4096)
- [ ] Password authentication disabled
- [ ] SFTP-only access (no shell access)
- [ ] Chroot jail configured
- [ ] File permissions properly set (700/600)
- [ ] Firewall rules limiting SFTP port access
- [ ] Logging enabled for all SFTP transactions
- [ ] Regular key rotation policy implemented
- [ ] Known hosts verification enabled

## Method 2: VPN Configuration

### OpenVPN Setup (Recommended)

#### Server Installation

```bash
# Install OpenVPN and Easy-RSA
sudo apt-get update
sudo apt-get install openvpn easy-rsa

# Set up PKI infrastructure
make-cadir ~/openvpn-ca
cd ~/openvpn-ca
```

#### Configure Certificate Authority

```bash
# Edit vars file
nano vars

# Set these values:
export KEY_COUNTRY="US"
export KEY_PROVINCE="MA"
export KEY_CITY="Boston"
export KEY_ORG="YourHospital"
export KEY_EMAIL="admin@yourhospital.org"
export KEY_OU="MedicalIT"

# Build CA
source vars
./clean-all
./build-ca

# Build server certificate
./build-key-server aimedres-vpn-server

# Generate Diffie-Hellman parameters
./build-dh

# Generate HMAC signature
openvpn --genkey --secret keys/ta.key
```

#### OpenVPN Server Configuration

Create `/etc/openvpn/server.conf`:

```conf
# Network settings
port 1194
proto udp
dev tun

# Certificates
ca /etc/openvpn/ca.crt
cert /etc/openvpn/aimedres-vpn-server.crt
key /etc/openvpn/aimedres-vpn-server.key
dh /etc/openvpn/dh2048.pem
tls-auth /etc/openvpn/ta.key 0

# Network configuration
server 10.8.0.0 255.255.255.0
push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 8.8.8.8"
push "dhcp-option DNS 8.8.4.4"

# Security settings
cipher AES-256-CBC
auth SHA256
tls-version-min 1.2

# Connection settings
keepalive 10 120
persist-key
persist-tun

# Logging
status /var/log/openvpn-status.log
log-append /var/log/openvpn.log
verb 3

# Compression
comp-lzo

# User/group privileges
user nobody
group nogroup
```

#### Client Configuration Template

Create client configuration file:

```conf
client
dev tun
proto udp

# Server address
remote aimedres-vpn.hospital.org 1194

# Security
cipher AES-256-CBC
auth SHA256
tls-version-min 1.2

# Connection
resolv-retry infinite
nobind
persist-key
persist-tun

# Certificates (inline or path)
ca ca.crt
cert client.crt
key client.key
tls-auth ta.key 1

# Compression
comp-lzo

# Logging
verb 3
```

### VPN Security Checklist

- [ ] Strong encryption (AES-256-CBC minimum)
- [ ] TLS 1.2+ enforced
- [ ] Certificate-based authentication
- [ ] HMAC authentication enabled
- [ ] Regular certificate rotation
- [ ] Split tunneling disabled (for maximum security)
- [ ] Connection logs monitored
- [ ] Failed connection attempts tracked
- [ ] Client certificates revoked when needed

## Method 3: Secure REST APIs

### API Security Configuration

#### 1. Enable HTTPS/TLS

```python
from flask import Flask
import ssl

app = Flask(__name__)

# SSL context for production
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(
    certfile='/etc/ssl/certs/aimedres.crt',
    keyfile='/etc/ssl/private/aimedres.key'
)

# Enforce TLS 1.2+
context.minimum_version = ssl.TLSVersion.TLSv1_2

# Run with HTTPS
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=443,
        ssl_context=context
    )
```

#### 2. API Authentication

```python
from src.aimedres.security.auth import SecureAuthManager
from flask import request, jsonify
import functools

auth_manager = SecureAuthManager({
    'jwt_secret': os.getenv('JWT_SECRET_KEY'),
    'token_expiry_hours': 24
})

def require_auth(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Validate token
        if not auth_manager.validate_token(token):
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

# Protected endpoint
@app.route('/api/v1/secure/data', methods=['POST'])
@require_auth
def receive_secure_data():
    data = request.get_json()
    # Process data
    return jsonify({'status': 'success'})
```

#### 3. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

@app.route('/api/v1/data/upload', methods=['POST'])
@limiter.limit("10 per minute")
@require_auth
def upload_data():
    # Handle upload
    pass
```

### API Client Example

```python
import requests
import json
from typing import Dict, Any

class SecureAPIClient:
    """Secure API client for AiMedRes"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'AiMedRes-Client/1.0'
        })
        
        # Verify SSL certificates
        self.session.verify = True
    
    def upload_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload data to AiMedRes API"""
        try:
            response = self.session.post(
                f'{self.base_url}/api/v1/data/upload',
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload failed: {e}")
            raise
    
    def get_results(self, request_id: str) -> Dict[str, Any]:
        """Retrieve results from AiMedRes API"""
        try:
            response = self.session.get(
                f'{self.base_url}/api/v1/results/{request_id}',
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Get results failed: {e}")
            raise

# Usage
client = SecureAPIClient(
    base_url='https://aimedres.hospital.org',
    api_key=os.getenv('AIMEDRES_API_KEY')
)

# Upload de-identified data
result = client.upload_data({
    'patient_id': 'HASHED_ID_12345',
    'data': {...}
})
```

### API Security Checklist

- [ ] HTTPS/TLS 1.2+ enforced
- [ ] API key or JWT authentication required
- [ ] Rate limiting configured
- [ ] Input validation enabled
- [ ] CORS properly configured
- [ ] Request/response logging enabled
- [ ] PHI scrubber integrated at API boundary
- [ ] Error messages don't leak sensitive info
- [ ] Regular security header checks (HSTS, CSP, etc.)

## Transfer Method Comparison

| Method | Use Case | Security Level | Setup Complexity | Real-time Capability |
|--------|----------|----------------|------------------|----------------------|
| SFTP | Batch file transfers | High | Medium | No |
| VPN | Network-level access | Very High | High | Yes |
| REST API | Real-time integration | High | Low-Medium | Yes |

## Firewall Configuration

### For SFTP

```bash
# Allow SFTP (SSH) port
sudo ufw allow 22/tcp

# Or custom SFTP port
sudo ufw allow 2222/tcp

# Limit to specific IP ranges
sudo ufw allow from 192.168.1.0/24 to any port 22
```

### For VPN

```bash
# Allow OpenVPN port
sudo ufw allow 1194/udp

# Allow from specific IPs only (recommended)
sudo ufw allow from 203.0.113.0/24 to any port 1194 proto udp
```

### For HTTPS API

```bash
# Allow HTTPS
sudo ufw allow 443/tcp

# Restrict to known IPs
sudo ufw allow from 192.168.1.0/24 to any port 443
```

## Monitoring and Logging

### Log Configuration

```bash
# Create log directory
sudo mkdir -p /var/log/aimedres/transfers
sudo chown aimedres:aimedres /var/log/aimedres/transfers
sudo chmod 750 /var/log/aimedres/transfers

# Configure logrotate
sudo nano /etc/logrotate.d/aimedres-transfers
```

Add:

```conf
/var/log/aimedres/transfers/*.log {
    daily
    rotate 90
    compress
    delaycompress
    notifempty
    create 640 aimedres aimedres
    sharedscripts
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}
```

### Transfer Audit Script

```python
import logging
from datetime import datetime

# Configure transfer audit logging
transfer_logger = logging.getLogger('aimedres.transfers')
transfer_logger.setLevel(logging.INFO)

handler = logging.FileHandler('/var/log/aimedres/transfers/audit.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
transfer_logger.addHandler(handler)

def log_transfer(method, direction, filename, size, success, user=None):
    """Log data transfer for audit trail"""
    transfer_logger.info(
        f"Transfer: method={method}, direction={direction}, "
        f"file={filename}, size={size}, success={success}, user={user}"
    )
```

## Compliance Requirements

### HIPAA Compliance

All transfer methods must ensure:

1. **Encryption in Transit**: TLS 1.2+ or SSH encryption
2. **Authentication**: Strong authentication (keys, certificates)
3. **Access Controls**: Principle of least privilege
4. **Audit Logging**: Complete audit trail of all transfers
5. **Data Integrity**: Verification of transferred data
6. **Transmission Security**: Secure protocols only

### Validation Script

```python
def validate_transfer_security():
    """Validate that transfer methods meet security requirements"""
    checks = {
        'SFTP_configured': check_sftp_config(),
        'SSH_keys_secure': check_ssh_keys(),
        'TLS_version_min': check_tls_version(),
        'Audit_logging_enabled': check_audit_logging(),
        'Firewall_configured': check_firewall_rules(),
    }
    
    all_passed = all(checks.values())
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")
    
    return all_passed
```

## Troubleshooting

### SFTP Connection Issues

```bash
# Test SFTP connection
sftp -vvv -i ~/.ssh/aimedres_sftp_key aimedres-sftp@server

# Check SSH logs
sudo tail -f /var/log/auth.log
```

### VPN Connection Issues

```bash
# Check OpenVPN status
sudo systemctl status openvpn@server

# View OpenVPN logs
sudo tail -f /var/log/openvpn.log
```

### API Connection Issues

```bash
# Test API connectivity
curl -v https://aimedres.hospital.org/health

# Check SSL certificate
openssl s_client -connect aimedres.hospital.org:443 -showcerts
```

## References

- HIPAA Security Rule: [HHS.gov](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- NIST Cryptographic Standards: [NIST.gov](https://csrc.nist.gov/)
- OpenSSH Documentation: [OpenSSH.com](https://www.openssh.com/manual.html)
