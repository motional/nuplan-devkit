FROM python:3.9.7-slim-buster

RUN apt-get update \
    && apt-get install -y curl gnupg2 software-properties-common default-jdk

RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    && apt-get update \
    && apt-get install -y \
        bazel \
        file \
        zip \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
