#!/bin/bash
HOME_DIR=./
cd ${HOME_DIR}
export EMBED_DIR=data/bert-embedding/
export DATA_DIR=data/ner/

lr=1e-5
ep=300
b=20
s=128
wp=0.1
run=LAST_TIME
weiNA=0.1
weiA=0.38


CUDA_VISIBLE_DEVICES=0 python src/main.py \
  --data_dir ${DATA_DIR} \
  --bert_dir ${EMBED_DIR} \
  --bert_file bert-base-cased \
  --do_train \
  --do_eval \
  --do_test \
  --WeightsA ${weiA} \
  --WeightsNA ${weiNA} \
  --learning_rate ${lr} \
  --epoch ${ep} \
  --use_cuda \
  --batch_size ${b} \
  --max_seq_length ${s} \
  --warmup_propotion ${wp} \
  --output_dir ruosenresults/ruosende/ner_output_lr${lr}_ep${ep}_b${b}_s${s}_wp${wp}_run${run}_weiNA${weiNA}_weiA${weiA}/ \
  --shell_print shell \
  --suffix last
# --do_eval \
#  --multi_gpu
#  --do_eval \
