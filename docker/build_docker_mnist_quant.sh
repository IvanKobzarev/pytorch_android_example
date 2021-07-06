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

export COMMAND='git fetch --all && git reset --hard origin/master 2>&1'
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

### Build pytorch android for trained model
################################################################################

export COMMAND='bash ./model/build_local_pytorch_for_mnist_quant.sh 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh

docker cp ./command.sh $id:${DOCKER_WORKDIR}

docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"
################################################################################


docker cp \
  $id:$DOCKER_WORKDIR/third_party/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar \
  $ROOT/android/application/app/aars/pytorch_android_quant.aar

docker commit $id $DOCKER_IMAGE

docker stop $id

