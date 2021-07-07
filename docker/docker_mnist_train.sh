#!/bin/bash
set -ex

DOCKER_IMAGE="pytorch_android_example"
ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
DOCKER_WORKDIR=/usr/local/pytorch_android_example

id=$(docker run -t -d -w ${DOCKER_WORKDIR} ${DOCKER_IMAGE})

### XXX FIXME:
docker cp \
  $ROOT/pytorch-patches/build_pytorch_android.sh \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/scripts/build_pytorch_android.sh
### XXX

export COMMAND='git fetch origin && git reset --hard origin/gh/ivankobzarev/36/orig 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh
docker cp ./command.sh $id:${DOCKER_WORKDIR}
docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"

### Train model
################################################################################

export COMMAND='python model/mnist.py 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh
docker cp ./command.sh $id:${DOCKER_WORKDIR}
docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"

docker exec -i -w ${DOCKER_WORKDIR}/model/output ${id} ls

for f in mnist.pt mnist.ptl mnist_ops.yaml mnist_quantized.pt mnist_quantized.ptl mnist_quantized_ops.yaml
do
  docker cp $id:${DOCKER_WORKDIR}/model/output/$f $ROOT/model/output/
done

cp $ROOT/model/output/mnist.ptl $ROOT/android/application/app/src/main/assets/
cp $ROOT/model/output/mnist_quantized.ptl $ROOT/android/application/app/src/main/assets/

docker stop $id

