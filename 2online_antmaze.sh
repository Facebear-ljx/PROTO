#!/bin/bash

for seed in 1 2 3 4 5; do
ENV='antmaze'
for DATASET in 'large-diverse' 'medium-play' 'medium-diverse' 'large-play'; do
ENV_NAME=$ENV'-'$DATASET'-v2'

python train_finetune.py \
 --tau_actor=0.00005 \
 --decay_speed=0.9 \
 --env_name=$ENV_NAME \
 --seed=$seed \
 --temp=0.5 \
 --num_pretraining_steps=200000 \
 --eval_episodes=100 \
 --eval_interval=50000 \
 --entropy_backup=False \
 --double_online=False \
 --config=configs/antmaze_finetune_config.py \
 --symmetric=True \
 --min_temp_online=0.0
# tau_actor and decay_speed is the hyperparameter for offline-to-online.

sleep 5
done
done


#  nohup python train_finetune.py \
#  --tau_actor=0.00005 \
#  --decay_speed=1 \
#  --env_name=antmaze-large-play-v2 \
#  --seed=0 \
#  --temp=0.5 \
#  --num_pretraining_steps=200000 \
#  --eval_episodes=100 \
#  --eval_interval=50000 \
#  --entropy_backup=False \
#  --double_online=False \
#  --config=configs/antmaze_finetune_config.py \
#  --symmetric=False &