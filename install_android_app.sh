#!/bin/bash
set -ex
ROOT="$( cd "$(dirname "$0")"; pwd -P)"

gradle -p $ROOT/android/application/ app:installDebug
