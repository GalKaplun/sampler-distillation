#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}

python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[noisy_cifar]' \
  logger.project=ensembling \
  datamodule.add_noise=True $OTHER_PARAMS
