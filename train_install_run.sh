#!/bin/bash
sh ./train_model_and_build_pytorch_in_docker.sh && sh ./install_android_app.sh && adb shell am start -n org.pytorch.demo.mnist/.MainActivity

