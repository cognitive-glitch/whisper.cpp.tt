FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN cmake -B build -DGGML_TTNN=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j

ENTRYPOINT ["/bin/bash"]
