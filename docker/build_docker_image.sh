#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

docker build -t pytorch_android_example - < $SCRIPTPATH/Dockerfile
