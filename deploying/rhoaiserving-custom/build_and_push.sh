#!/bin/bash

podman login -u=developer -p=developer registry.ocp4.example.com:8443
podman build . -t  registry.ocp4.example.com:8443/developer/sklearn-model-server
podman push registry.ocp4.example.com:8443/developer/sklearn-model-server
