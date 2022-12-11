#! /bin/bash

WORKING_DIR=/opt/data/private/nlp03/kdwang/dialog_projects/zzz_eva

DATASET_NAME="kdconv"
DATA_PATH="${WORKING_DIR}/data/${DATASET_NAME}"  # path of the directory of the dataset
CACHE_PATH="${WORKING_DIR}/cache/${DATASET_NAME}"

MODEL_SIZE="base"  # base large xlarge
CONFIG_PATH="/opt/data/private/nlp03/kdwang/huggingface_models/EVA2-${MODEL_SIZE}"
MODEL_PATH="/opt/data/private/nlp03/kdwang/huggingface_models/EVA2-${MODEL_SIZE}"
TOKENIZER_PATH="/opt/data/private/nlp03/kdwang/huggingface_models/EVA2-${MODEL_SIZE}"  # vocab path

FINETUNE_SAVE_PATH="${WORKING_DIR}/src/save_models/finetune"

TRAIN_STEPS=100000
VALID_STEP_NUM=180
LOSS_STEP_NUM=120
BATCH_SIZE=2
EPOCHS=10
ENC_LEN=128 # max input length of encoder
DEC_LEN=128 # max input length of decoder
ENC_KG_LEN=512 # max input kg length of encoder
GPU_NUM=0

PROJECTS_NAME="12-7-test"
SAVE_MODEL_PATH=${PROJECTS_NAME}

# 日志参数
LOG_FILE_NAME="${PROJECTS_NAME}.log"
TRAIN_LOGS_PATH="${WORKING_DIR}/src/logs/train_logs/${MODEL_SIZE}_model"  # 训练记录文件夹
TRAIN_LOG_FILE="${TRAIN_LOGS_PATH}/${LOG_FILE_NAME}"  # ============此次train的log=============

OPTS=""
# 模型参数
OPTS+=" --pretrain-model-path ${MODEL_PATH}"
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --train_from_middle_epoch 0"
OPTS+=" --gpu-index ${GPU_NUM}"
OPTS+=" --model-size ${MODEL_SIZE}"
OPTS+=" --save-model-path ${SAVE_MODEL_PATH}"  # 保存的模型的名字
# 训练参数
OPTS+=" --valid_early_stop_num 10"
OPTS+=" --train_loss_early_stop_epoch 2"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --lr 0.0002"  # 使用上面的变量有bug
OPTS+=" --save-finetune ${FINETUNE_SAVE_PATH}/${MODEL_SIZE}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --do-eval"
OPTS+=" --eval-generation" # run the evaluation of generation
OPTS+=" --warmup 0.00"
OPTS+=" --loss-step-num ${LOSS_STEP_NUM}"
OPTS+=" --valid-step-num ${VALID_STEP_NUM}"
# log参数
OPTS+=" --train-logs-path ${TRAIN_LOGS_PATH}"
OPTS+=" --train-log-file ${TRAIN_LOG_FILE}"
# data参数
OPTS+=" --data-path ${DATA_PATH}"  # data数据所在文件夹
OPTS+=" --cache-path ${CACHE_PATH}"  # data的cache数据 所在文件夹 
OPTS+=" --enc-seq-length ${ENC_LEN}"  # Context长度
OPTS+=" --dec-seq-length ${DEC_LEN}"  # Decoder长度
OPTS+=" --enc-kg-length ${ENC_KG_LEN}"  # 外部知识长度
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"  # tokenizer

# echo ${OPTS}

/opt/data/private/nlp03/kdwang/anaconda3/envs/eva/bin/python ${WORKING_DIR}/src/eva_finetuning.py ${OPTS}

