#/bin/sh
set -e

ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "ROOT:$ROOT"

PYTORCH_ROOT=$ROOT/third_party/pytorch
echo "PYTORCH_ROOT:$PYTORCH_ROOT"

pushd $PYTORCH_ROOT
#TODO: Do we need to verify that submodules are init recursive? (common error)
git submodule update --init --recursive

SELECTED_OP_LIST=$ROOT/model/output/mnist_ops.yaml \
  sh ./scripts/build_pytorch_android.sh x86

ln -s \
  $ROOT/third_party/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar \
  $ROOT/android/application/app/aars/pytorch_android_fp32.aar


popd

