# Network Security Configuration Guide

## Overview

This guide provides comprehensive instructions for implementing network security measures for AiMedRes deployment, ensuring HTTPS/TLS enforcement, firewall configuration, and network isolation.

## Network Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Internet/WAN                        │
└───────────────────┬─────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │   Firewall/WAF      │
         │  (TLS Termination)  │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │      DMZ Zone       │
         │   (Load Balancer)   │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   Application Zone  │
         │   (AiMedRes API)    │
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │    Database Zone    │
         │   (Secure Backend)  │
         └─────────────────────┘
```

## HTTPS/TLS Configuration

### Requirement

**All traffic must use HTTPS/TLS 1.2 or higher:**
- API endpoints
- Web UI
- File transfers
- Inter-service communication

### SSL/TLS Certificate Setup

#### 1. Obtain SSL Certificate

**Option A: Let's Encrypt (Free, Automated)**

```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d aimedres.hospital.org -d api.aimedres.hospital.org

# Verify auto-renewal
sudo certbot renew --dry-run
```

**Option B: Commercial Certificate**

```bash
# Generate CSR
openssl req -new -newkey rsa:4096 -nodes \
  -keyout aimedres.key \
  -out aimedres.csr \
  -subj "/C=US/ST=MA/L=Boston/O=Hospital/CN=aimedres.hospital.org"

# Submit CSR to CA and receive certificate
# Place certificate files in /etc/ssl/certs/
```

**Option C: Internal CA (For Development/Testing)**

```bash
# Generate self-signed certificate (NOT for production)
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout /etc/ssl/private/aimedres.key \
  -out /etc/ssl/certs/aimedres.crt \
  -days 365 \
  -subj "/C=US/ST=MA/L=Boston/O=Hospital/CN=aimedres.hospital.org"
```

#### 2. Configure Nginx with TLS

Create `/etc/nginx/sites-available/aimedres`:

```nginx
# HTTP to HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name aimedres.hospital.org api.aimedres.hospital.org;
    
    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

# HTTPS server configuration
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name aimedres.hospital.org;
    
    # SSL Certificate Configuration
    ssl_certificate /etc/ssl/certs/aimedres.crt;
    ssl_certificate_key /etc/ssl/private/aimedres.key;
    
    # SSL Protocol Configuration (TLS 1.2+)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    
    # Strong Cipher Suites (HIPAA Compliant)
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    
    # SSL Session Configuration
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-chain.crt;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Logging
    access_log /var/log/nginx/aimedres_access.log;
    error_log /var/log/nginx/aimedres_error.log;
    
    # Proxy to AiMedRes API
    location /api/ {
        proxy_pass http://127.0.0.1:8002/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files (if serving UI)
    location / {
        root /var/www/aimedres;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://127.0.0.1:8002/health;
        access_log off;
    }
}

# API subdomain
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.aimedres.hospital.org;
    
    ssl_certificate /etc/ssl/certs/aimedres.crt;
    ssl_certificate_key /etc/ssl/private/aimedres.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
    
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    
    location / {
        proxy_pass http://127.0.0.1:8002/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration:

```bash
# Test configuration
sudo nginx -t

# Enable site
sudo ln -s /etc/nginx/sites-available/aimedres /etc/nginx/sites-enabled/

# Reload Nginx
sudo systemctl reload nginx
```

#### 3. Flask Application TLS Configuration

For running Flask directly with TLS (development/testing):

```python
from flask import Flask
import ssl

app = Flask(__name__)

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.minimum_version = ssl.TLSVersion.TLSv1_2
context.maximum_version = ssl.TLSVersion.TLSv1_3

# Load certificates
context.load_cert_chain(
    certfile='/etc/ssl/certs/aimedres.crt',
    keyfile='/etc/ssl/private/aimedres.key'
)

# Configure cipher suites
context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS')

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=443,
        ssl_context=context
    )
```

For production with Gunicorn:

```bash
# gunicorn_config.py
bind = '0.0.0.0:443'
workers = 4
worker_class = 'sync'
timeout = 60

# SSL Configuration
certfile = '/etc/ssl/certs/aimedres.crt'
keyfile = '/etc/ssl/private/aimedres.key'
ssl_version = 'TLSv1_2'  # Minimum version
ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
```

Run with:

```bash
gunicorn --config gunicorn_config.py main:app
```

### TLS Verification Script

```python
import ssl
import socket
from datetime import datetime

def verify_tls_configuration(hostname, port=443):
    """Verify TLS configuration meets security requirements"""
    
    print(f"🔍 Verifying TLS configuration for {hostname}:{port}")
    
    context = ssl.create_default_context()
    
    try:
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Get certificate info
                cert = ssock.getpeercert()
                cipher = ssock.cipher()
                version = ssock.version()
                
                # Check TLS version
                print(f"\n📋 TLS Configuration:")
                print(f"  Protocol: {version}")
                
                if version in ['TLSv1.2', 'TLSv1.3']:
                    print(f"  ✅ TLS version acceptable")
                else:
                    print(f"  ❌ TLS version too old: {version}")
                    return False
                
                # Check cipher
                print(f"\n🔐 Cipher Suite:")
                print(f"  Cipher: {cipher[0]}")
                print(f"  Protocol: {cipher[1]}")
                print(f"  Bits: {cipher[2]}")
                
                # Check certificate expiration
                print(f"\n📜 Certificate:")
                print(f"  Subject: {cert['subject']}")
                print(f"  Issuer: {cert['issuer']}")
                
                not_after = datetime.strptime(
                    cert['notAfter'],
                    '%b %d %H:%M:%S %Y %Z'
                )
                days_until_expiry = (not_after - datetime.now()).days
                
                print(f"  Expires: {cert['notAfter']} ({days_until_expiry} days)")
                
                if days_until_expiry < 30:
                    print(f"  ⚠️  Certificate expires soon!")
                elif days_until_expiry < 0:
                    print(f"  ❌ Certificate expired!")
                    return False
                else:
                    print(f"  ✅ Certificate valid")
                
                print(f"\n✅ TLS configuration verified successfully")
                return True
                
    except ssl.SSLError as e:
        print(f"❌ SSL/TLS error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

# Usage
if __name__ == '__main__':
    verify_tls_configuration('aimedres.hospital.org')
```

## Firewall Configuration

### UFW (Uncomplicated Firewall) Setup

```bash
# Reset firewall (if needed)
sudo ufw --force reset

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port if non-standard)
sudo ufw allow 22/tcp
sudo ufw limit 22/tcp  # Rate limit SSH

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow HTTP (for redirect to HTTPS)
sudo ufw allow 80/tcp

# Allow from specific IP ranges only (recommended)
# Hospital network
sudo ufw allow from 192.168.1.0/24 to any port 443 proto tcp

# EMR/EHR system
sudo ufw allow from 10.0.50.0/24 to any port 443 proto tcp

# Database access (internal only)
sudo ufw allow from 127.0.0.1 to any port 5432 proto tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### iptables Configuration (Advanced)

```bash
#!/bin/bash
# /etc/iptables/rules.v4

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (with rate limiting)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTPS from specific networks
iptables -A INPUT -p tcp -s 192.168.1.0/24 --dport 443 -j ACCEPT
iptables -A INPUT -p tcp -s 10.0.50.0/24 --dport 443 -j ACCEPT

# Allow HTTP (for redirect)
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow ICMP (ping) from local network
iptables -A INPUT -p icmp -s 192.168.1.0/24 -j ACCEPT

# Log dropped packets (for monitoring)
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables INPUT denied: " --log-level 7

# Save rules
iptables-save > /etc/iptables/rules.v4
```

Make persistent:

```bash
# Install iptables-persistent
sudo apt-get install iptables-persistent

# Save current rules
sudo netfilter-persistent save

# Restore on boot
sudo systemctl enable netfilter-persistent
```

### Firewall Verification Script

```python
import subprocess
import re

def verify_firewall_rules():
    """Verify firewall configuration"""
    
    print("🔍 Verifying firewall configuration...")
    
    try:
        # Check UFW status
        result = subprocess.run(
            ['sudo', 'ufw', 'status', 'verbose'],
            capture_output=True,
            text=True
        )
        
        output = result.stdout
        
        # Check if firewall is active
        if 'Status: active' not in output:
            print("❌ Firewall is not active!")
            return False
        else:
            print("✅ Firewall is active")
        
        # Check for required rules
        required_rules = [
            (r'443/tcp.*ALLOW', 'HTTPS port'),
            (r'80/tcp.*ALLOW', 'HTTP redirect port'),
        ]
        
        for pattern, description in required_rules:
            if re.search(pattern, output):
                print(f"✅ {description} rule found")
            else:
                print(f"❌ {description} rule missing")
                return False
        
        # Check default policies
        if 'Default: deny (incoming)' in output:
            print("✅ Default incoming policy: DENY")
        else:
            print("❌ Default incoming policy not set to DENY")
            return False
        
        print("\n✅ Firewall configuration verified")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying firewall: {e}")
        return False

if __name__ == '__main__':
    verify_firewall_rules()
```

## Network Isolation and Segmentation

### VLAN Configuration

Separate network segments for different components:

```
VLAN 10 - Management Network (192.168.10.0/24)
VLAN 20 - Application Network (192.168.20.0/24)
VLAN 30 - Database Network (192.168.30.0/24)
VLAN 40 - DMZ Network (192.168.40.0/24)
```

### Docker Network Isolation

```yaml
# docker-compose.yml with network isolation
version: '3.8'

networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  backend:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.20.1.0/24
  database:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.20.2.0/24

services:
  nginx:
    image: nginx:latest
    networks:
      - frontend
    ports:
      - "443:443"
      - "80:80"
  
  aimedres-api:
    build: .
    networks:
      - frontend
      - backend
    # No port exposure to host
  
  postgres:
    image: postgres:15
    networks:
      - database
    # No port exposure to host
```

### Network Access Control Lists (ACLs)

```python
# Network ACL configuration for different segments
NETWORK_ACLS = {
    'dmz': {
        'allowed_sources': [
            '0.0.0.0/0',  # Internet
        ],
        'allowed_destinations': [
            '192.168.20.0/24',  # Application network
        ],
        'allowed_ports': [80, 443]
    },
    'application': {
        'allowed_sources': [
            '192.168.40.0/24',  # DMZ
            '192.168.10.0/24',  # Management
        ],
        'allowed_destinations': [
            '192.168.30.0/24',  # Database
        ],
        'allowed_ports': [5432, 6379]  # PostgreSQL, Redis
    },
    'database': {
        'allowed_sources': [
            '192.168.20.0/24',  # Application only
        ],
        'allowed_destinations': [],
        'allowed_ports': [5432]
    }
}
```

## DDoS Protection

### Rate Limiting with Nginx

```nginx
# Define rate limit zones
http {
    # Limit requests per IP
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login_limit:10m rate=5r/m;
    
    # Limit connections per IP
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    
    server {
        # Apply rate limiting
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            limit_conn conn_limit 10;
            proxy_pass http://backend;
        }
        
        location /api/auth/login {
            limit_req zone=login_limit burst=5 nodelay;
            proxy_pass http://backend;
        }
    }
}
```

### Fail2ban Configuration

```bash
# Install fail2ban
sudo apt-get install fail2ban

# Create filter for AiMedRes
sudo nano /etc/fail2ban/filter.d/aimedres.conf
```

Add:

```ini
[Definition]
failregex = ^<HOST> .* "POST /api/auth/login HTTP.*" 401
            ^<HOST> .* "POST /api/.* HTTP.*" 403
ignoreregex =
```

Configure jail:

```bash
sudo nano /etc/fail2ban/jail.local
```

Add:

```ini
[aimedres]
enabled = true
port = http,https
filter = aimedres
logpath = /var/log/nginx/aimedres_access.log
maxretry = 5
findtime = 600
bantime = 3600
action = iptables-multiport[name=aimedres, port="http,https", protocol=tcp]
```

Restart fail2ban:

```bash
sudo systemctl restart fail2ban
sudo fail2ban-client status aimedres
```

## Monitoring and Alerting

### Network Monitoring Script

```python
import psutil
import logging
from datetime import datetime

logger = logging.getLogger('network_monitor')

def monitor_network_traffic():
    """Monitor network traffic and connections"""
    
    # Get network I/O statistics
    net_io = psutil.net_io_counters()
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'bytes_sent': net_io.bytes_sent,
        'bytes_recv': net_io.bytes_recv,
        'packets_sent': net_io.packets_sent,
        'packets_recv': net_io.packets_recv,
        'errors_in': net_io.errin,
        'errors_out': net_io.errout,
        'drops_in': net_io.dropin,
        'drops_out': net_io.dropout
    }
    
    # Get active connections
    connections = psutil.net_connections(kind='inet')
    
    stats['active_connections'] = len([
        c for c in connections
        if c.status == 'ESTABLISHED'
    ])
    
    stats['listening_ports'] = len([
        c for c in connections
        if c.status == 'LISTEN'
    ])
    
    # Check for suspicious activity
    if stats['errors_in'] > 100 or stats['drops_in'] > 100:
        logger.warning(f"High network errors detected: {stats}")
    
    if stats['active_connections'] > 1000:
        logger.warning(f"High connection count: {stats['active_connections']}")
    
    return stats

# Usage
if __name__ == '__main__':
    import time
    
    while True:
        stats = monitor_network_traffic()
        print(f"Network Stats: {stats}")
        time.sleep(60)
```

## Security Checklist

### Network Security Validation

- [ ] TLS 1.2+ enforced on all endpoints
- [ ] SSL certificates valid and not expiring soon
- [ ] HTTP redirects to HTTPS
- [ ] Strong cipher suites configured
- [ ] HSTS header enabled
- [ ] Security headers configured (CSP, X-Frame-Options, etc.)
- [ ] Firewall enabled and configured
- [ ] Default deny policy set
- [ ] Only required ports open
- [ ] Network segmentation implemented
- [ ] DMZ configured for external-facing services
- [ ] Database network isolated
- [ ] Rate limiting configured
- [ ] DDoS protection enabled
- [ ] Fail2ban configured
- [ ] Network monitoring active
- [ ] Intrusion detection system (IDS) considered
- [ ] Regular security audits scheduled

## Compliance Verification

```python
def verify_network_compliance():
    """Comprehensive network security compliance check"""
    
    checks = {
        'TLS Configuration': verify_tls_configuration('aimedres.hospital.org'),
        'Firewall Active': verify_firewall_rules(),
        'Rate Limiting': check_rate_limiting(),
        'Network Isolation': check_network_isolation(),
    }
    
    print("\n📋 Network Security Compliance Report")
    print("=" * 60)
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n✅ All network security checks passed")
    else:
        print("\n❌ Some network security checks failed")
    
    return all_passed

if __name__ == '__main__':
    verify_network_compliance()
```

## References

- NIST SP 800-52: [Guidelines for TLS Implementation](https://csrc.nist.gov/publications/detail/sp/800-52/rev-2/final)
- HIPAA Security Rule: [Network Security Standards](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- Mozilla SSL Configuration Generator: [ssl-config.mozilla.org](https://ssl-config.mozilla.org/)
