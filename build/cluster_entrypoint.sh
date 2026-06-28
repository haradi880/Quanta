#!/usr/bin/env bash
set -euo pipefail

for variable in RAY_TLS_SERVER_CERT RAY_TLS_SERVER_KEY RAY_TLS_CA_CERT; do
  [[ -f "${!variable:-}" ]] || {
    echo "missing required mTLS file: $variable" >&2
    exit 3
  }
done

python3.11 -m uvicorn cluster.node_server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-certfile "$RAY_TLS_SERVER_CERT" \
  --ssl-keyfile "$RAY_TLS_SERVER_KEY" \
  --ssl-ca-certs "$RAY_TLS_CA_CERT" \
  --ssl-cert-reqs 2 &
NODE_SERVER_PID=$!
trap 'kill "$NODE_SERVER_PID" 2>/dev/null || true' EXIT

case "${HARADIBOTS_CLUSTER_ROLE:-worker}" in
  head)
    ray start --head --node-ip-address=0.0.0.0 --port=6379 \
      --dashboard-host=0.0.0.0
    exec python3.11 -m uvicorn cluster.api_server:app --host 0.0.0.0 --port 8000
    ;;
  worker)
    exec ray start --address="${HARADIBOTS_RAY_HEAD:-head:6379}" --block
    ;;
  *)
    echo "invalid HARADIBOTS_CLUSTER_ROLE" >&2
    exit 2
    ;;
esac
