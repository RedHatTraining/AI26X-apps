#!/usr/bin/env bash

IMAGE=quay.io/redhattraining/ai264-xgboost-notebook:v2-20240319

podman build . -t ${IMAGE}

podman push ${IMAGE}