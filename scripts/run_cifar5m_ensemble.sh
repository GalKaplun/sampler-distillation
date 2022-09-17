#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}

python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[cifar5m-5-ensemble]' \
  logger.project=ensembling \
  datamodule.dataset=cifar5m \
  trainer.max_epochs=1 \
  model.teach_arch=resnet18 \
  model.teach_dir=./models/cifar10/noisy \
  model.max_teachers=5 \
  model.ensemble=True $OTHER_PARAMS
