#!/bin/bash

ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
echo "ROOT:$ROOT"

python $ROOT/model/mnist.py --skip-training
