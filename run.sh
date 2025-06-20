## M
online_learning='full'
i=1
ns=(1 )
bszs=(1 )
lens=(24 48)
methods=('DMCnet' ) 
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do

CUDA_VISIBLE_DEVICES=1  nohup python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data PCpower --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 3 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  pcpower$len$m$online_learning$m.out 2>&1 &
CUDA_VISIBLE_DEVICES=1  nohup python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data ETTh2 --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 3 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  ETTh2$len$m$online_learning.out 2>&1 &
CUDA_VISIBLE_DEVICES=1  nohup python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data ETTm1 --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 3 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  ETTm1$len$m$online_learning.out 2>&1 &
CUDA_VISIBLE_DEVICES=1  nohup python -u main.py  --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data WTH --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 3 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  WTH$len$m$online_learning.out 2>&1 &
CUDA_VISIBLE_DEVICES=1  nohup python -u main.py --method $m --root_path ./data/ --n_inner $n --test_bsz $bsz --data Weather --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 3 --learning_rate 1e-3 --online_learning $online_learning --use_adbfgs >  Weather$len$m$online_learning.out 2>&1 &

done
done
done
done










