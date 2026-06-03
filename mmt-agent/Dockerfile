# mmt-agent image — MMT (probe/dpi/security) + the agent, baked into Ubuntu 20.04.
#
# Building on a pinned 20.04 base makes the edge host's own OS irrelevant: the
# exact runtime libraries MMT needs live inside the image. Confirmed via `ldd`
# on a working host, the only non-base libs required are libpcap0.8, libxml2,
# libicu66.
#
# Build context layout (assembled by the SSH provisioner):
#   Dockerfile, entrypoint.sh, packages/*.deb, agent/<agent source>
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libs for MMT + node + openssl (self-signed cert) + setcap.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpcap0.8 libxml2 libicu66 libcap2-bin openssl ca-certificates curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# mmt-probe's postinst runs `systemctl daemon-reload`, which fails in a build
# container (no systemd) and aborts the install. Stub systemctl to exit 0 in
# every standard bin dir so dpkg's maintainer scripts find it. The container
# never uses systemd (the agent runs via the entrypoint), so this is safe.
RUN for d in /usr/local/sbin /usr/local/bin /usr/sbin /usr/bin /sbin /bin; do \
      printf '#!/bin/sh\nexit 0\n' > "$d/systemctl" && chmod +x "$d/systemctl"; \
    done

# Install MMT. Use `apt-get install <local .deb>` rather than `dpkg -i` so any
# missing dependency is resolved automatically and install order is handled.
COPY packages/ /tmp/mmt/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       /tmp/mmt/mmt-dpi_*.deb /tmp/mmt/mmt-probe_*.deb /tmp/mmt/mmt-security_*.deb \
    && ldconfig \
    && rm -rf /tmp/mmt /var/lib/apt/lists/*

# Grant capture capabilities so the probe can sniff without sudo
# (the container is run with --cap-add=NET_RAW,NET_ADMIN).
RUN setcap cap_net_raw,cap_net_admin+eip "$(command -v mmt-probe)" || true

# Agent source.
WORKDIR /opt/mmt-agent
COPY agent/ /opt/mmt-agent/
RUN npm install --omit=dev --no-audit --no-fund

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV AGENT_WORK_DIR=/var/lib/mmt-agent \
    AGENT_PORT=8443 \
    USE_SUDO=false
EXPOSE 8443

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
