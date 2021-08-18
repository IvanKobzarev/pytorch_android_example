# PyTorch Mobile Android Example

## Quickstart
1. Build Docker image `pytorch_android_example`
```
sh ./build_docker_image.sh
```
Docker image is based on ubuntu and contains all dependencies to build pytorch for linux and pytorch for android, including Vulkan and NNAPI backends.

Result:
```
└─ $ docker image ls
REPOSITORY                TAG       IMAGE ID       CREATED        SIZE
pytorch_android_example   latest    8cc67601109c   21 hours ago   12.4GB
```

2. Run model train and building pytorch android
```
sh ./train_model_and_build_pytorch_in_docker.sh
```
Result:
```
└─ $ ls model/output/
mnist-nnapi-ops.yaml	mnist-ops-all.yaml	mnist-quant.pt		mnist-vulkan.pt		mnist.ptl
mnist-nnapi.pt		mnist-ops.yaml		mnist-quant.ptl		mnist-vulkan.ptl
mnist-nnapi.ptl		mnist-quant-ops.yaml	mnist-vulkan-ops.yaml	mnist.pt
└─ $ ls android/application/app/aars/
pytorch_android.aar
```
3. Install android application on connected android device
```
sh ./install_android_app.sh
```
Result:
Application `MNIST` installed.

## Content
`docker/Dockerfile` - docker container definition 

`model/mnist.py` - MNIST model definition and training code. Serializes the model and quantized variant of it for `CPU`, `Vulkan` and `NNAPI`. 

`pytorch-patches` - temporary changes in pytorch that will be eliminated after fixing functionality in pytorch master

`sh ./model/build_local_pytorch_for_mnist.sh` - runs build of pytorch android, contains specification of android abis for the build.






