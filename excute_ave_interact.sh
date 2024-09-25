python train_interact_ave_spike_net_amp.py --model_scale XL --train_data AVE_dataset --dataset_path ./AVE_dataset --t_step 6 --epoch 90 --n_mels 96  --batch_size 16 --pool_kern_size 16 --lr 1e-4 --fusion_method concat > ./interact_ave_log/interact_ave_concat_xl.log 2>&1

python train_interact_ave_spike_net_amp.py --model_scale XL --train_data AVE_dataset --dataset_path ./AVE_dataset --t_step 6 --epoch 90 --n_mels 96  --batch_size 16 --pool_kern_size 16 --lr 1e-4 --fusion_method sum > ./interact_ave_log/interact_ave_sum_xl.log 2>&1

python train_interact_ave_spike_net_amp.py --model_scale XL --train_data AVE_dataset --dataset_path ./AVE_dataset --t_step 6 --epoch 90 --n_mels 96  --batch_size 16 --pool_kern_size 16 --lr 1e-4 --fusion_method ew > ./interact_ave_log/interact_ave_ew_xl.log 2>&1
