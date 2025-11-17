#!/bin/bash
#
# Traffic Switching Script for Blue/Green Deployment
# Updates load balancer or reverse proxy configuration to route traffic
#

set -e

TARGET_ENV=$1

if [ -z "$TARGET_ENV" ]; then
    echo "Usage: $0 <blue|green>"
    exit 1
fi

# Configuration
NGINX_CONF="${NGINX_CONF:-/etc/nginx/sites-available/aimedres.conf}"
NGINX_ENABLED="${NGINX_ENABLED:-/etc/nginx/sites-enabled/aimedres.conf}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Update Nginx upstream configuration
update_nginx() {
    local target_port=8002
    
    if [ "$TARGET_ENV" == "green" ]; then
        target_port=8003
    fi
    
    log "Updating Nginx configuration for $TARGET_ENV (port $target_port)..."
    
    # Backup current configuration
    if [ -f "$NGINX_CONF" ]; then
        cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Update upstream backend
    cat > "$NGINX_CONF" << EOF
upstream aimedres_backend {
    server localhost:${target_port};
}

server {
    listen 80;
    server_name aimedres.hospital.org;
    
    # Redirect HTTP to HTTPS
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aimedres.hospital.org;
    
    ssl_certificate /etc/ssl/certs/aimedres.crt;
    ssl_certificate_key /etc/ssl/private/aimedres.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://aimedres_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /health {
        proxy_pass http://aimedres_backend/health;
        access_log off;
    }
}
EOF
    
    # Enable site if not already enabled
    if [ ! -L "$NGINX_ENABLED" ]; then
        ln -sf "$NGINX_CONF" "$NGINX_ENABLED"
    fi
    
    # Test configuration
    if nginx -t 2>&1 | grep -q "successful"; then
        log "Nginx configuration test passed"
        
        # Reload Nginx
        systemctl reload nginx || service nginx reload
        
        if [ $? -eq 0 ]; then
            log "Traffic switched to $TARGET_ENV successfully"
            return 0
        else
            echo "ERROR: Failed to reload Nginx"
            return 1
        fi
    else
        echo "ERROR: Nginx configuration test failed"
        # Restore backup
        if [ -f "${NGINX_CONF}.backup.$(date +%Y%m%d_%H%M%S)" ]; then
            mv "${NGINX_CONF}.backup.$(date +%Y%m%d_%H%M%S)" "$NGINX_CONF"
        fi
        return 1
    fi
}

# Alternative: Update HAProxy configuration
update_haproxy() {
    log "HAProxy switching not implemented. Please configure manually."
    return 0
}

# Main function
main() {
    log "Switching traffic to $TARGET_ENV environment..."
    
    # Check which load balancer is in use
    if command -v nginx &> /dev/null && [ -f "$NGINX_CONF" ]; then
        update_nginx
    elif command -v haproxy &> /dev/null; then
        update_haproxy
    else
        log "WARNING: No supported load balancer found (Nginx or HAProxy)"
        log "Please manually update your load balancer to point to $TARGET_ENV"
        return 0
    fi
}

main "$@"
