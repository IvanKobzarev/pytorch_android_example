#!/bin/bash
set -ex

DOCKER_IMAGE="pytorch_android_example"
ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
DOCKER_WORKDIR=/usr/local/pytorch_android_example

id=$(docker run -t -d -w ${DOCKER_WORKDIR} ${DOCKER_IMAGE})

export COMMAND='((sh ./model/build_local_pytorch_for_mnist_fp32.sh) | docker exec -i "$id" bash) 2>&1'

echo ${COMMAND} > ./command.sh && bash ./command.sh

docker cp \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar \
  $ROOT/android/application/app/aars/pytorch_android_fp32.aar

docker commit "$id" ${DOCKER_IMAGE}_commit


