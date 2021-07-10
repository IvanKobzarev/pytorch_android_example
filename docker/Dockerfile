FROM ubuntu:latest

RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install curl
RUN apt-get -y install wget
RUN apt-get -y install unzip
RUN pip3 install --upgrade pip
RUN pip install pyyaml
RUN pip install Pillow
RUN apt-get install -y git
RUN apt-get -y install python3.8-dev
RUN apt-get -y install moreutils

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cmake protobuf-compiler
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3

# Install OpenJDK
# https://hub.docker.com/r/picoded/ubuntu-openjdk-8-jdk/dockerfile/

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

# Fix certificate issues, found as of
# https://bugs.launchpad.net/ubuntu/+source/ca-certificates-java/+bug/983302

RUN apt-get update && \
    apt-get install -y ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
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

ARG cmdline_tools=https://dl.google.com/android/repository/commandlinetools-linux-6609375_latest.zip
ARG android_home=/opt/android/sdk


RUN apt-get update && \
    apt-get install --yes \
        xvfb lib32z1 lib32stdc++6 build-essential \
        libcurl4-openssl-dev libglu1-mesa libxi-dev libxmu-dev \
        libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*


# Download and install Android Commandline Tools
RUN mkdir -p ${android_home}/cmdline-tools && \
    wget -O /tmp/cmdline-tools.zip -t 5 "${cmdline_tools}" && \
    unzip -q /tmp/cmdline-tools.zip -d ${android_home}/cmdline-tools && \
    rm /tmp/cmdline-tools.zip

# Set environmental variables
# deprecated upstream, should be removed in next-gen image
ENV ANDROID_HOME ${android_home}
ENV ANDROID_SDK_ROOT ${android_home}
ENV ADB_INSTALL_TIMEOUT 120
ENV PATH=${ANDROID_SDK_ROOT}/emulator:${ANDROID_SDK_ROOT}/cmdline-tools/tools/bin:${ANDROID_SDK_ROOT}/tools:${ANDROID_SDK_ROOT}/tools/bin:${ANDROID_SDK_ROOT}/platform-tools:${PATH}

RUN mkdir ~/.android && echo '### User Sources for Android SDK Manager' > ~/.android/repositories.cfg

RUN yes | sdkmanager --licenses && yes | sdkmanager --update

# Update SDK manager and install system image, platform and build tools
RUN sdkmanager \
  "tools" \
  "platform-tools" \
  "emulator"

RUN sdkmanager \
  "build-tools;28.0.3" \
  "build-tools;30.0.3"

RUN sdkmanager "platforms;android-28"

ARG gradle_home=/opt/gradle
ARG gradle_version=6.8.3
RUN rm -rf ${gradle_home} && mkdir -p ${gradle_home}

RUN wget --no-verbose --output-document=/tmp/gradle.zip \
  "https://services.gradle.org/distributions/gradle-${gradle_version}-bin.zip" && \
  unzip -q /tmp/gradle.zip -d ${gradle_home} && rm /tmp/gradle.zip

ENV GRADLE_HOME ${gradle_home}
ENV PATH=${gradle_home}/bin/:${PATH}

RUN mkdir -p ~/.ssh
RUN ssh-keyscan -t rsa github.com > ~/.ssh/known_hosts

WORKDIR /usr/local
RUN git clone https://github.com/IvanKobzarev/pytorch_android_example.git

WORKDIR /usr/local/pytorch_android_example
RUN git submodule update --init --recursive

RUN bash ./docker/install_vulkan_sdk.sh

ENV VULKAN_SDK=/usr/local/vulkansdk

WORKDIR /usr/local/pytorch_android_example/third_party/pytorch
RUN pip install -r requirements.txt
RUN USE_VULKAN=1 USE_NNAPI=1 python setup.py install --cmake

RUN pip3 install --no-deps torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


