# Cluster mTLS material

Run `bash build/gen_certs.sh --node <node-id>` during secure provisioning.
Private keys and generated certificates are ignored and must be mounted as
secrets. No node may join without a certificate signed by the cluster CA.
