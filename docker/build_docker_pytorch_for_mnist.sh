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

docker cp \
  $ROOT/pytorch-patches/unpickler.cpp \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/torch/csrc/jit/serialization/unpickler.cpp

docker cp \
  $ROOT/pytorch-patches/op_allowlist.h \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/aten/src/ATen/core/op_registration/op_allowlist.h
### XXX

export COMMAND='git fetch --all && git reset --hard origin/master 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh
docker cp ./command.sh $id:${DOCKER_WORKDIR}
docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"

### XXX FIXME:
docker cp \
  $ROOT/pytorch-patches/build_pytorch_android.sh \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/scripts/build_pytorch_android.sh

docker cp \
  $ROOT/pytorch-patches/op_allowlist.h \
  $id:${DOCKER_WORKDIR}/third_party/pytorch/aten/src/ATen/core/op_registration/op_allowlist.h
### XXX

### Train model
################################################################################

export COMMAND='python model/mnist.py --skip-training 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh
docker cp ./command.sh $id:${DOCKER_WORKDIR}
docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"

docker exec -i -w ${DOCKER_WORKDIR}/model/output ${id} ls

# Fp32
for f in mnist.pt mnist.ptl mnist-ops.yaml
do
  docker cp $id:${DOCKER_WORKDIR}/model/output/$f $ROOT/model/output/
done
cp $ROOT/model/output/mnist.ptl $ROOT/android/application/app/src/main/assets/

# Quant
for f in mnist-quant.pt mnist-quant.ptl mnist-quant-ops.yaml
do
  docker cp $id:${DOCKER_WORKDIR}/model/output/$f $ROOT/model/output/
done
cp $ROOT/model/output/mnist-quant.ptl $ROOT/android/application/app/src/main/assets/

# NNAPI
#for f in mnist-nnapi.pt mnist-nnapi.ptl mnist-nnapi-ops.yaml
#do
#  docker cp $id:${DOCKER_WORKDIR}/model/output/$f $ROOT/model/output/
#done
#cp $ROOT/model/output/mnist-nnapi.ptl $ROOT/android/application/app/src/main/assets/

# Vulkan
for f in mnist-vulkan.pt mnist-vulkan.ptl mnist-vulkan-ops.yaml
do
  docker cp $id:${DOCKER_WORKDIR}/model/output/$f $ROOT/model/output/
done
cp $ROOT/model/output/mnist-vulkan.ptl $ROOT/android/application/app/src/main/assets/
cp $ROOT/model/output/mnist-vulkan.pt $ROOT/android/application/app/src/main/assets/

docker cp $id:${DOCKER_WORKDIR}/model/output/mnist-ops-all.yaml $ROOT/model/output/




### Build pytorch android for trained model
################################################################################

export COMMAND='bash ./model/build_local_pytorch_for_mnist.sh 2>&1'
echo ${COMMAND} > ./command.sh
chmod 755 ./command.sh

docker cp ./command.sh $id:${DOCKER_WORKDIR}

docker exec -i -w ${DOCKER_WORKDIR} ${id} sh "${DOCKER_WORKDIR}/command.sh"
################################################################################


docker cp \
  $id:$DOCKER_WORKDIR/third_party/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar \
  $ROOT/android/application/app/aars/pytorch_android.aar

docker commit $id $DOCKER_IMAGE

docker stop $id

