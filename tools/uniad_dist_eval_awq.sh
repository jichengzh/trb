#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28598}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# ------------- 解决 datasets 包名冲突：把 site-packages 放最前 --------------- #
HF_SITE=$(python - <<'PY'
import site, sys, pathlib
print(next(p for p in site.getsitepackages() if pathlib.Path(p).exists()))
PY
)
export PYTHONPATH="${HF_SITE}:$(dirname $0)/..:$PYTHONPATH"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/jcz_uniad_awq.py \
    $CFG \
    $CKPT \
    ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T