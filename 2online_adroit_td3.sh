#!/bin/bash

for seed in 1 2 3 4 5; do
for ENV in 'pen' 'hammer' 'door'; do
for lmbda in 2.5; do
DATASET='human'
ENV_NAME=$ENV'-'$DATASET'-v0'

python train_finetune_td3.py \
 --tau_actor=0.00005 \
 --decay_speed=0.9 \
 --env_name=$ENV_NAME \
 --seed=$seed \
 --temp=2 \
 --lmbda=$lmbda \
 --num_pretraining_steps=100000 \
 --eval_episodes=10 \
 --eval_interval=5000 \
 --entropy_backup=False \
 --double_online=False \
 --config=configs/adroit_config.py \
 --min_temp_online=0.0 \
 --update_freq=2 \
 --symmetric=False \
# tau_actor and decay_speed is the hyperparameter for offline-to-online.

sleep 5
done
done
done