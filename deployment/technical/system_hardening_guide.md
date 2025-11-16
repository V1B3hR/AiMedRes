# AiMedRes System Hardening Guide

## Purpose
This guide provides comprehensive system hardening procedures to secure AiMedRes deployments in healthcare environments, following CIS benchmarks, NIST guidelines, and healthcare-specific security requirements.

---

## 1. Operating System Hardening

### 1.1 Ubuntu/Debian Systems

#### Apply Security Updates
```bash
# Update package lists
sudo apt update

# Upgrade all packages
sudo apt upgrade -y

# Enable automatic security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

#### Kernel Hardening
Edit `/etc/sysctl.conf` and add:
```bash
# IP Forwarding (disable if not needed)
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2

# IP spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP ping requests (optional)
# net.ipv4.icmp_echo_ignore_all = 1

# Kernel address space randomization
kernel.randomize_va_space = 2
```

Apply settings:
```bash
sudo sysctl -p
```

#### Disable Unnecessary Services
```bash
# List all running services
systemctl list-unit-files --state=enabled

# Disable unnecessary services (examples)
sudo systemctl disable bluetooth.service
sudo systemctl disable cups.service
sudo systemctl disable avahi-daemon.service
```

#### Configure Secure Shell (SSH)
Edit `/etc/ssh/sshd_config`:
```
# Change default port (security through obscurity)
Port 2222

# Disable root login
PermitRootLogin no

# Use SSH protocol 2 only
Protocol 2

# Limit authentication attempts
MaxAuthTries 3
MaxSessions 2

# Enable public key authentication
PubkeyAuthentication yes

# Disable password authentication (after setting up keys)
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no

# Disable X11 forwarding
X11Forwarding no

# Set login grace time
LoginGraceTime 30

# Limit user access
AllowUsers aimedres admin

# Use strong ciphers and MACs
Ciphers aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com,hmac-sha2-512,hmac-sha2-256
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org,diffie-hellman-group-exchange-sha256
```

Restart SSH:
```bash
sudo systemctl restart sshd
```

---

## 2. User and Access Control

### 2.1 Principle of Least Privilege

#### Create Dedicated Service Account
```bash
# Create aimedres user with no login shell
sudo useradd -r -s /bin/false aimedres

# Create aimedres group
sudo groupadd aimedres

# Add user to group
sudo usermod -a -G aimedres aimedres
```

#### Configure File Permissions
```bash
# Application directory
sudo chown -R aimedres:aimedres /opt/aimedres
sudo chmod -R 750 /opt/aimedres

# Configuration files (read-only for application)
sudo chmod 640 /opt/aimedres/config.yml
sudo chmod 600 /opt/aimedres/.env

# Log directory
sudo mkdir -p /var/log/aimedres
sudo chown -R aimedres:aimedres /var/log/aimedres
sudo chmod 750 /var/log/aimedres

# Data directory (with encryption)
sudo mkdir -p /var/lib/aimedres/data
sudo chown -R aimedres:aimedres /var/lib/aimedres
sudo chmod 700 /var/lib/aimedres/data
```

### 2.2 sudo Configuration
Create `/etc/sudoers.d/aimedres`:
```
# Allow specific commands for aimedres admin
aimedres_admin ALL=(aimedres) NOPASSWD: /usr/bin/systemctl restart aimedres
aimedres_admin ALL=(aimedres) NOPASSWD: /usr/bin/systemctl status aimedres
```

### 2.3 Password Policy
Edit `/etc/login.defs`:
```
PASS_MAX_DAYS   90
PASS_MIN_DAYS   1
PASS_MIN_LEN    14
PASS_WARN_AGE   7
```

Install and configure PAM:
```bash
sudo apt install libpam-pwquality -y
```

Edit `/etc/security/pwquality.conf`:
```
minlen = 14
dcredit = -1
ucredit = -1
ocredit = -1
lcredit = -1
minclass = 4
maxrepeat = 2
```

---

## 3. Network Security

### 3.1 Firewall Configuration (UFW)

```bash
# Install UFW
sudo apt install ufw -y

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (use custom port if changed)
sudo ufw allow 2222/tcp

# Allow HTTPS
sudo ufw allow 443/tcp

# Allow application ports (only from trusted networks)
sudo ufw allow from 10.0.0.0/8 to any port 8000 proto tcp
sudo ufw allow from 10.0.0.0/8 to any port 8001 proto tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### 3.2 Advanced Firewall Rules (iptables)

For more granular control:
```bash
# Flush existing rules
sudo iptables -F

# Default policies
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
sudo iptables -A INPUT -p tcp --dport 2222 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 2222 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
sudo iptables -A INPUT -p tcp --dport 2222 -j ACCEPT

# Allow HTTPS
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application from trusted network
sudo iptables -A INPUT -p tcp -s 10.0.0.0/8 --dport 8000 -j ACCEPT

# Log dropped packets
sudo iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables_INPUT_denied: " --log-level 7

# Save rules
sudo apt install iptables-persistent -y
sudo netfilter-persistent save
```

### 3.3 Network Segmentation

Place AiMedRes in a dedicated VLAN:
- Application servers: VLAN 10 (10.10.10.0/24)
- Database servers: VLAN 20 (10.10.20.0/24)
- Storage: VLAN 30 (10.10.30.0/24)
- Management: VLAN 100 (10.10.100.0/24)

Configure ACLs on network switches to restrict traffic between VLANs.

---

## 4. Docker Container Hardening

### 4.1 Docker Daemon Configuration

Create/edit `/etc/docker/daemon.json`:
```json
{
  "icc": false,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp.json",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
```

### 4.2 Container Security Best Practices

#### Run as Non-Root User
Already implemented in Dockerfile:
```dockerfile
USER aimedres
```

#### Resource Limits
In docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

#### Read-Only Root Filesystem (where possible)
```yaml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
  - /run
```

#### Drop Unnecessary Capabilities
```yaml
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
```

#### Use Security Profiles
```yaml
security_opt:
  - apparmor=docker-default
  - seccomp=/path/to/seccomp/profile.json
```

---

## 5. Data Encryption

### 5.1 Encryption at Rest

#### Encrypted File System (LUKS)
```bash
# Create encrypted partition
sudo cryptsetup luksFormat /dev/sdb1

# Open encrypted partition
sudo cryptsetup luksOpen /dev/sdb1 aimedres_data

# Create filesystem
sudo mkfs.ext4 /dev/mapper/aimedres_data

# Mount
sudo mount /dev/mapper/aimedres_data /var/lib/aimedres/data

# Add to /etc/crypttab for auto-mount
echo "aimedres_data /dev/sdb1 none luks" | sudo tee -a /etc/crypttab

# Add to /etc/fstab
echo "/dev/mapper/aimedres_data /var/lib/aimedres/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

#### Database Encryption
PostgreSQL TDE (Transparent Data Encryption):
```sql
-- Enable encryption for tablespace
CREATE TABLESPACE encrypted_data
  LOCATION '/var/lib/postgresql/encrypted'
  WITH (encryption = 'aes-256-gcm');

-- Create encrypted table
CREATE TABLE patient_data (
  ...
) TABLESPACE encrypted_data;
```

### 5.2 Encryption in Transit

#### TLS/SSL Configuration
Generate certificates:
```bash
# Self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt or organizational CA
sudo apt install certbot -y
sudo certbot certonly --standalone -d aimedres.example.com
```

Configure TLS in application:
```yaml
# config.yml
security:
  tls:
    enabled: true
    cert_file: /etc/ssl/certs/aimedres-cert.pem
    key_file: /etc/ssl/private/aimedres-key.pem
    min_version: "1.3"
    ciphers:
      - TLS_AES_256_GCM_SHA384
      - TLS_CHACHA20_POLY1305_SHA256
      - TLS_AES_128_GCM_SHA256
```

---

## 6. Logging and Monitoring

### 6.1 Centralized Logging

#### Configure rsyslog
Edit `/etc/rsyslog.d/aimedres.conf`:
```
# AiMedRes application logs
if $programname == 'aimedres' then /var/log/aimedres/application.log
if $programname == 'aimedres' and $syslogseverity <= 3 then /var/log/aimedres/error.log

# Audit logs
if $programname == 'aimedres-audit' then /var/log/aimedres/audit.log

# Security logs
if $programname == 'aimedres-security' then /var/log/aimedres/security.log

# Stop processing
& stop
```

#### Forward to SIEM
```
# Forward to central log server
*.* @@siem.example.com:514
```

### 6.2 Audit Logging (auditd)

```bash
# Install auditd
sudo apt install auditd audispd-plugins -y

# Configure audit rules
sudo vim /etc/audit/rules.d/aimedres.rules
```

Add rules:
```
# Monitor configuration changes
-w /opt/aimedres/config.yml -p wa -k aimedres_config_change

# Monitor data access
-w /var/lib/aimedres/data/ -p r -k aimedres_data_access

# Monitor executable
-w /opt/aimedres/bin/ -p x -k aimedres_execution

# Monitor authentication
-w /var/log/aimedres/auth.log -p wa -k aimedres_auth
```

Reload rules:
```bash
sudo augenrules --load
sudo systemctl restart auditd
```

---

## 7. Intrusion Detection

### 7.1 OSSEC HIDS

```bash
# Install OSSEC
wget https://github.com/ossec/ossec-hids/archive/refs/tags/3.7.0.tar.gz
tar -xzf 3.7.0.tar.gz
cd ossec-hids-3.7.0
sudo ./install.sh

# Configure for local installation
# Select: local
# Enable: syscheck, rootcheck, active-response
```

Configure monitored files in `/var/ossec/etc/ossec.conf`:
```xml
<directories check_all="yes" realtime="yes">
  /opt/aimedres/config.yml
</directories>
<directories check_all="yes" realtime="yes">
  /var/lib/aimedres/data
</directories>
```

### 7.2 Fail2ban

```bash
# Install fail2ban
sudo apt install fail2ban -y

# Create jail for AiMedRes
sudo vim /etc/fail2ban/jail.d/aimedres.conf
```

Add configuration:
```
[aimedres-auth]
enabled = true
port = 8000,8001
filter = aimedres-auth
logpath = /var/log/aimedres/auth.log
maxretry = 3
bantime = 3600
findtime = 600
```

Create filter:
```bash
sudo vim /etc/fail2ban/filter.d/aimedres-auth.conf
```

```
[Definition]
failregex = ^.*Authentication failed for user.*from <HOST>.*$
            ^.*Invalid API key from <HOST>.*$
ignoreregex =
```

---

## 8. Compliance Hardening

### 8.1 HIPAA Technical Safeguards

- [ ] Unique user identification implemented
- [ ] Emergency access procedures documented
- [ ] Automatic logoff configured (15 minutes idle)
- [ ] Encryption and decryption enabled
- [ ] Audit controls activated
- [ ] Integrity controls implemented
- [ ] Person or entity authentication enabled
- [ ] Transmission security configured

### 8.2 CIS Benchmark Compliance

Run CIS-CAT assessment:
```bash
# Download CIS-CAT from https://www.cisecurity.org/
# Run assessment
./CIS-CAT.sh -a -r /path/to/benchmark.xml
```

Key benchmarks to verify:
- File system permissions
- Service configuration
- Network parameters
- Logging and auditing
- Authentication and authorization

---

## 9. Backup and Recovery Hardening

### 9.1 Secure Backup Configuration

```bash
# Create backup script with encryption
cat > /opt/aimedres/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/aimedres"
DATE=$(date +%Y%m%d_%H%M%S)
PASSPHRASE="your-secure-passphrase"

# Backup application data
tar -czf - /var/lib/aimedres/data | \
  gpg --symmetric --cipher-algo AES256 --passphrase "$PASSPHRASE" \
  > "$BACKUP_DIR/aimedres_data_$DATE.tar.gz.gpg"

# Backup database
PGPASSWORD="$DB_PASSWORD" pg_dump -U aimedres -h localhost aimedres | \
  gpg --symmetric --cipher-algo AES256 --passphrase "$PASSPHRASE" \
  > "$BACKUP_DIR/aimedres_db_$DATE.sql.gpg"

# Rotate old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gpg" -mtime +30 -delete
EOF

chmod 700 /opt/aimedres/backup.sh
```

### 9.2 Test Restore Procedures

Document and test restore procedures quarterly.

---

## 10. Security Checklist

### Pre-Deployment Security Verification

- [ ] OS patched and up-to-date
- [ ] Kernel hardening parameters applied
- [ ] Unnecessary services disabled
- [ ] SSH hardened and key-based auth configured
- [ ] Firewall configured and tested
- [ ] Docker daemon hardened
- [ ] Containers running as non-root
- [ ] Data encryption at rest enabled
- [ ] TLS/SSL configured for all communications
- [ ] Audit logging enabled and tested
- [ ] Intrusion detection installed and configured
- [ ] Backup and restore tested
- [ ] Security scanning completed (vulnerability assessment)
- [ ] Penetration testing performed (if required)
- [ ] Incident response plan documented
- [ ] Security policies and procedures documented

---

## 11. Ongoing Maintenance

### Monthly Tasks
- [ ] Review security logs
- [ ] Apply security patches
- [ ] Review user accounts and permissions
- [ ] Test backup and restore
- [ ] Review firewall rules

### Quarterly Tasks
- [ ] Comprehensive security audit
- [ ] Update security documentation
- [ ] Vulnerability scanning
- [ ] Review and update incident response plan
- [ ] Security awareness training

### Annual Tasks
- [ ] Full penetration testing
- [ ] Compliance audit (HIPAA, etc.)
- [ ] Disaster recovery drill
- [ ] Security policy review and update

---

## 12. References

- **CIS Benchmarks**: https://www.cisecurity.org/cis-benchmarks/
- **NIST Cybersecurity Framework**: https://www.nist.gov/cyberframework
- **HIPAA Security Rule**: https://www.hhs.gov/hipaa/for-professionals/security/
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **Docker Security**: https://docs.docker.com/engine/security/

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Next Review**: Quarterly  
**Owner**: Security Team

---

**IMPORTANT**: This hardening guide should be reviewed and customized for your specific environment and threat model. Work with your security team to ensure all measures are appropriate and properly implemented.
