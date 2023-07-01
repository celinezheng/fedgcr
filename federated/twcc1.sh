
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode afl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode drfl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode drfl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode term --test --batch 64



# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedmix --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ablation --test --batch 64

# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode afl --test --batch 64
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode q-ffl --test --batch 64

# python train.py --dataset domainnet --percent 0.5 --expname even --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fpl --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fpl --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ablation --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51 --lr 0.0001
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode ablation --test --batch 64

# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedsam --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedsam --test --batch 64



# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode afl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode term --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode harmo-fl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode harmo-fl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedmix --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode fedsam --test --batch 64

# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode afl --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode term --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode harmo-fl --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedmix --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode fedsam --test --batch 64


python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedavg --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode afl --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode q-ffl --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode term --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode harmo-fl --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedmix --test --batch 64
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.4 --expname uneven --mode fedsam --test --batch 64

python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedavg --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode afl --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode q-ffl --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode term --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode harmo-fl --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedmix --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --ratio 1.5 --expname uneven --mode fedsam --test --batch 64

# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"