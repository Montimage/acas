#!/bin/sh
# Container entrypoint: ensure a self-signed TLS cert exists, then run the agent.
# ACAS connects with insecureTLS for the self-signed cert (trusted operator LAN).
set -e

CERT_DIR="${AGENT_WORK_DIR:-/var/lib/mmt-agent}/tls"
if [ ! -f "$CERT_DIR/cert.pem" ]; then
  mkdir -p "$CERT_DIR"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=mmt-agent" >/dev/null 2>&1
fi

export AGENT_TLS_CERT="$CERT_DIR/cert.pem"
export AGENT_TLS_KEY="$CERT_DIR/key.pem"

exec node /opt/mmt-agent/server.js
