#!/bin/bash

for seed in 1 2 3 4 5; do
for BC_PRETRAIN in False; do
for tau_actor in 0.005; do
for lmbda in 2.5; do
for ENV in 'halfcheetah' 'hopper' 'walker2d'; do
for DATASET in 'medium' 'medium-replay' 'random'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
echo $ENV_NAME
echo $BC_PRETRAIN
python train_finetune_td3.py \
 --tau_actor=$tau_actor \
 --decay_speed=0.9 \
 --env_name=$ENV_NAME \
 --seed=$seed \
 --temp=2 \
 --num_pretraining_steps=100000 \
 --eval_episodes=10 \
 --eval_interval=5000 \
 --entropy_backup=True \
 --double_online=True \
 --config=configs/mujoco_config_finetune.py \
 --bc_pretrain=$BC_PRETRAIN \
 --symmetric=False \
 --min_temp_online=0.0 \
 --lmbda=$lmbda \
# tau_actor and decay_speed is the hyperparameter for offline-to-online.

sleep 5
done
done
done
done
done
done

#
# nohup python train_finetune.py \
#  --tau_actor=0.00 \
#  --decay_speed=0.5 \
#  --env_name=hopper-medium-v2 \
#  --seed=3 \
#  --temp=2 \
#  --num_pretraining_steps=100000 \
#  --eval_episodes=10 \
#  --eval_interval=5000 \
#  --entropy_backup=True \
#  --double_online=True \
#  --config=configs/mujoco_config_finetune.py \
#  --bc_pretrain=False \
#  --symmetric=True \
#  --min_temp_online=0.001 &