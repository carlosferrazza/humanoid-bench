#!/bin/sh
set -e
docker build -f agents/dreamerv3/Dockerfile -t img .
docker run -it --rm --gpus all --privileged -v /dev:/dev -v ~/logdir:/logdir img \
  sh -c 'ldconfig; sh scripts/xvfb_run.sh python agents/dreamerv3/train.py \
    --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
    --configs crafter'
