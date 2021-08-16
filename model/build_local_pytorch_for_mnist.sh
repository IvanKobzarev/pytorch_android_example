#/bin/sh
set -e

ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "ROOT:$ROOT"

CURRENT_DIR="$(pwd -P)"
PYTORCH_ROOT=$ROOT/third_party/pytorch
echo "PYTORCH_ROOT:$PYTORCH_ROOT"

cd $PYTORCH_ROOT

echo "***SELECTED_OP_LIST***"
cat $ROOT/model/output/mnist-ops-all.yaml
echo "======================"

#FIXME: build for all ABIs, remove x86
BUILD_LITE_INTERPRETER=0 \
SELECTED_OP_LIST=$ROOT/model/output/mnist-ops-all.yaml \
  sh ./scripts/build_pytorch_android.sh arm64-v8a

ln -sf \
  $ROOT/third_party/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar \
  $ROOT/android/application/app/aars/pytorch_android.aar

cd $CURRENT_DIR
