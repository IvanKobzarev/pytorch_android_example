#/bin/sh
set -e

ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "ROOT:$ROOT"

PYTORCH_ROOT=$ROOT/third_party/pytorch
echo "PYTORCH_ROOT:$PYTORCH_ROOT"

pushd $PYTORCH_ROOT

SELECTED_OP_LIST=$ROOT/model/output/mnist_quantized_ops.yaml sh ./scripts/build_pytorch_android.sh x86


popd

