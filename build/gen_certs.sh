#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HARADIBOTS_MTLS_DIR:-$ROOT/cluster/mtls}"
mkdir -p "$OUT"
if [[ -n "${MSYSTEM:-}" ]]; then
  # Prevent MSYS from rewriting OpenSSL subject strings such as /CN=...
  export MSYS2_ARG_CONV_EXCL="*"
fi

CA_KEY="$OUT/ca.key"
CA_CERT="$OUT/ca.crt"

if [[ ! -f "$CA_KEY" || ! -f "$CA_CERT" ]]; then
  openssl genrsa -out "$CA_KEY" 4096
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 \
    -subj "/CN=HaradiBots Cluster CA" -out "$CA_CERT"
fi

generate_node_cert() {
  local node_id="$1"
  [[ "$node_id" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "invalid node_id" >&2
    return 2
  }
  local key="$OUT/${node_id}.key"
  local csr="$OUT/${node_id}.csr"
  local cert="$OUT/${node_id}.crt"
  local ext="$OUT/${node_id}.ext"
  openssl genrsa -out "$key" 3072
  openssl req -new -key "$key" -subj "/CN=${node_id}" -out "$csr"
  printf 'extendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=DNS:%s\n' "$node_id" > "$ext"
  openssl x509 -req -in "$csr" -CA "$CA_CERT" -CAkey "$CA_KEY" \
    -CAcreateserial -out "$cert" -days 825 -sha256 -extfile "$ext"
  openssl verify -CAfile "$CA_CERT" "$cert"
  rm -f "$csr" "$ext"
}

if [[ "${1:-}" == "--node" ]]; then
  generate_node_cert "${2:?node id required}"
fi
