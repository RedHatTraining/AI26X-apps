#!/bin/bash
QUAY_HOST="registry.ocp4.example.com:8443"
podman login -u=developer -p=developer ${QUAY_HOST}
podman build . -t  ${QUAY_HOST}/developer/sklearn-model-server
podman push ${QUAY_HOST}/developer/sklearn-model-server

echo "Making the repository public..."
curl -s -X PUT -H "Authorization: Bearer $QUAY_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"visibility": "public"}' \
     https://${QUAY_HOST}/api/v1/repository/developer/sklearn-model-server
