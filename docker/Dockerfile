FROM ubuntu:latest

RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install curl
RUN apt-get -y install unzip
RUN apt-get -y install vim
RUN pip3 install --upgrade pip
RUN pip install pyyaml
RUN apt-get install -y git
RUN apt-get -y install python3.8-dev
RUN apt-get -y install expect
RUN apt-get -y install moreutils

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cmake protobuf-compiler
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3

ENV ANDROID_NDK /usr/local/android-ndk/
ENV ANDROID_NDK_VERSION r21d

ENV NDK_URL="https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip"

# Download Android NDK
RUN mkdir "$ANDROID_NDK" \
    && cd "$ANDROID_NDK" \
    && curl -C - -o ndk.zip $NDK_URL \
    && unzip ndk.zip \
    && rm ndk.zip

ENV ANDROID_NDK /usr/local/android-ndk/android-ndk-r21d

RUN mkdir -p ~/.ssh
RUN ssh-keyscan -t rsa github.com > ~/.ssh/known_hosts
WORKDIR /usr/local/
RUN git clone https://github.com/IvanKobzarev/pytorch_android_example.git

WORKDIR /usr/local/pytorch_android_example
RUN git checkout -b model-training origin/model-training

RUN git submodule update --init --recursive

