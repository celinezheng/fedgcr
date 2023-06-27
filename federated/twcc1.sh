

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode gifair --iters 2 --batch 64


# python train.py --dataset digit --percent 1 --ratio 1.5 --expname uneven --mode gifair --iters 50 --batch 64
# python train.py --dataset digit --percent 1 --ratio 1.5 --expname uneven --mode gifair --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop --iters 50 --batch 64 --pcon
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop --test --batch 64 --pcon


# python train.py --dataset digit --percent 1 --ratio 1.8 --expname uneven --mode gifair --iters 50 --batch 64
# python train.py --dataset digit --percent 1 --ratio 1.8 --expname uneven --mode gifair --test --batch 64

# python train.py --dataset digit --percent 1 --ratio 1.5 --expname uneven --mode propfair --iters 50 --batch 64
# python train.py --dataset digit --percent 1 --ratio 1.5 --expname uneven --mode propfair --test --batch 64

# python train.py --dataset digit --percent 1 --ratio 1.8 --expname uneven --mode propfair --iters 50 --batch 64
# python train.py --dataset digit --percent 1 --ratio 1.8 --expname uneven --mode propfair --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode propfair --iters 50 --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode propfair --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode propfair --iters 50 --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode propfair --test --batch 64


# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode ccop --iters 50 --batch 64 --pcon --test_freq 25
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode ccop --test --batch 64 --pcon
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode ccop --iters 50 --batch 64 --pcon --test_freq 25
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode ccop --test --batch 64 --pcon

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop --moon --iters 50 --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop --moon --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ccop --clscon --iters 50 --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ccop --clscon --test --batch 64

# python train.py --dataset digit --percent 0.5 --ratio 1.5 --test_freq 25 --expname uneven --mode ccop --clscon --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --test_freq 25 --expname uneven --mode ccop --clscon --test --batch 64

# python train.py --dataset digit --percent 0.5 --expname even --mode ccop --clscon --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --expname even --mode ccop --clscon --test --batch 64

# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode afl --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode afl --test --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode term --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode term --test --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode gifair --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode gifair --test --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode fedmix --iters 50 --batch 64
# python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode fedmix --test --batch 64
python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode propfair --iters 50 --batch 64
python train.py --dataset digit --percent 0.5 --test_freq 50 --expname even --mode propfair --test --batch 64

# python train.py --dataset domainnet --percent 0.1 --ratio 1.4 --expname uneven --mode propfair --iters 2 --batch 64


# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"