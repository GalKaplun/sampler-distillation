FROM pytorchlightning/pytorch_lightning:1.6.4-py3.9-torch1.10

RUN pip install --no-cache-dir --upgrade pip
RUN pip config set global.cache-dir /tmp
RUN pip install wandb
RUN pip install ipdb
RUN pip install numpy==1.21.0

WORKDIR /ensembling/
RUN mkdir -p ./.tmp/wandb/.config
ENV WANDB_DIR=./.tmp/wandb
ENV WANDB_CONFIG_DIR=./.tmp/wandb/.config

