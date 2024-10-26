FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh \
    && ln -s $CONDA_DIR/bin/conda /usr/local/bin/conda

ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -a -y

SHELL ["conda", "run", "-n", "adCNN", "/bin/bash", "-c"]

WORKDIR /app

COPY . .

RUN echo "source activate adCNN" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]

