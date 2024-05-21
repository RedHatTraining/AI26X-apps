#!/bin/bash
QUAY_HOST="registry.ocp4.example.com:8443"
podman login -u=developer -p=developer ${QUAY_HOST}
podman build . -t  ${QUAY_HOST}/developer/sklearn-model-server
podman push ${QUAY_HOST}/developer/sklearn-model-server
