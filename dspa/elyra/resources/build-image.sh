#!/usr/bin/env bash
# Build and push custom runtime image with statsmodels for Elyra pipelines.

set -euo pipefail

# Image registry and name
IMAGE_REGISTRY="${IMAGE_REGISTRY:-quay.io/redhattraining}"
IMAGE_NAME="dspa-elyra-runtime"
IMAGE_TAG="${IMAGE_TAG:-v2.0}"
IMAGE="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building image: ${IMAGE}"
podman build -f Containerfile -t "${IMAGE}" .

echo "Pushing image: ${IMAGE}"
podman push "${IMAGE}"

echo "Done. Update the pipeline template with the new image:"
echo "  ${IMAGE}"
