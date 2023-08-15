# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode afl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode drfl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode drfl --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode term --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode propfair --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode propfair --test --batch 64
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedmix --test --batch 64

# python train.py --dataset digit --percent 0.5 --expname even --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --expname even --mode fedmix --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fpl --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fpl --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedsam --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedsam --test --batch 64
