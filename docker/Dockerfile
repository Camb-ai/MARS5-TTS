FROM python:3

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/Camb-ai/MARS5-TTS"
LABEL org.opencontainers.image.licenses = "AGPL-3.0 license"


# enable passwordless ssh
RUN mkdir ~/.ssh && \
    printf "Host * \n    ForwardAgent yes\nHost *\n    StrictHostKeyChecking no" > ~/.ssh/config && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# enable RDMA support: this enables direct access to GPU memory
RUN apt-get update && \
    apt-get install -y infiniband-diags perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install MARS5-TTS
RUN git clone https://github.com/Camb-ai/MARS5-TTS.git \
    && cd ./MARS5-TTS \
    && pip install -r requirements.txt
