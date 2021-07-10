#!/bin/bash

set -ex

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_vulkansdk_dir=/usr/local/vulkansdk
_tmp_vulkansdk_targz=/tmp/vulkansdk.tar.gz

curl \
  --silent \
  --show-error \
  --location \
  --fail \
  --retry 3 \
  --output "${_tmp_vulkansdk_targz}" "https://sdk.lunarg.com/sdk/download/1.2.182.0/linux/vulkansdk-linux-x86_64-1.2.182.0.tar.gz"

mkdir -p "${_vulkansdk_dir}"
tar -C "${_vulkansdk_dir}" -xzf "${_tmp_vulkansdk_targz}" --strip-components 1
rm -rf "${_tmp_vulkansdk_targz}"
