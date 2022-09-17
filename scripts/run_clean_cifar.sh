#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}
export HYDRA_FULL_ERROR=1

python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[clean_cifar]' \
  logger.project=ensembling $OTHER_PARAMS
