python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode ccop --iters 50 --batch 64 --test_freq 51
python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode ccop --test --batch 64
python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode fedavg --test --batch 64
python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset office --percent 1 --ratio 1.7 --expname uneven --mode harmo-fl --test --batch 64

# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode drfl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode drfl --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode term --test --batch 64
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedmix --test --batch 64

# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode fedmix --test --batch 64

# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedmix --test --batch 64

# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode term --test --batch 64
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --shuffle --percent 0.5 --ratio 1.6 --expname uneven --mode fedmix --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fpl --test --batch 64


# kill

