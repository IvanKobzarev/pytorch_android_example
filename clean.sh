#!/bin/bash
set -ex
ROOT="$( cd "$(dirname "$0")"; pwd -P)"

rm -f $ROOT/android/application/app/aars/pytorch_android.aar

rm -f $ROOT/android/application/app/src/main/assets/*
