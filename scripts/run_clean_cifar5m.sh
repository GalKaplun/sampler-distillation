#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}

python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[cifar5m-no-teacher]' \
  logger.project=ensembling \
  datamodule.dataset=cifar5m  \
  trainer.max_epochs=1 $OTHER_PARAMS
