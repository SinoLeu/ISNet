#!/bin/bash


python train_interact_urban8k_spike_net_amp.py --model_scale XS --train_data UrbanSound8K-AV --dataset_path ./UrbanSound8K-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 16 --pool_kern_size 8 --lr 1e-3 --fusion_method concat > ./interact_urban8k_log/xsinteract_urban8k_concat_xs.log 2>&1

python train_interact_urban8k_spike_net_amp.py --model_scale XS --train_data UrbanSound8K-AV --dataset_path ./UrbanSound8K-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 16 --pool_kern_size 8 --lr 1e-3 --fusion_method sum > ./interact_urban8k_log/interact_urban8k_sum_xs.log 2>&1

python train_interact_urban8k_spike_net_amp.py --model_scale XS --train_data UrbanSound8K-AV --dataset_path ./UrbanSound8K-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 16 --pool_kern_size 8 --lr 1e-3 --fusion_method ew > ./interact_urban8k_log/interact_urban8k_ew_xs.log 2>&1

# python train_interact_urban8k_spike_net_amp.py --model_scale S --train_data UrbanSound8K-AV --dataset_path ./UrbanSound8K-AV --t_step 5 --epoch 30 --n_mels 64 --batch_size 16 --pool_kern_size 8 --lr 1e-3 --fusion_method att > ./interact_urban8k_log/interact_urban8k_att.log 2>&1




python train_interact_cifra_spike_net_amp.py --model_scale XS --train_data CIFAR10-AV --dataset_path ./CIFAR10-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 32 --pool_kern_size 8 --lr 1e-3 --fusion_method concat > ./interact_cifar10_log/interact_cifar10_concat_xs.log 2>&1

python train_interact_cifra_spike_net_amp.py --model_scale XS --train_data CIFAR10-AV --dataset_path ./CIFAR10-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 32 --pool_kern_size 8 --lr 1e-3 --fusion_method sum > ./interact_cifar10_log/interact_cifar10_sum_xs.log 2>&1

python train_interact_cifra_spike_net_amp.py --model_scale XS --train_data CIFAR10-AV --dataset_path ./CIFAR10-AV --t_step 5 --epoch 40 --n_mels 64 --batch_size 32 --pool_kern_size 8 --lr 1e-3 --fusion_method ew > ./interact_cifar10_log/interact_cifar10_ew_xs.log 2>&1