# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode afl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode drfl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode drfl --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode term --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode propfair --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode propfair --test --batch 64
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedmix --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedmix --test --batch 64

# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode fedavg --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode fedavg --test --batch 64
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode ablation --test --batch 64
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode afl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode afl --test --batch 64
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode q-ffl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode q-ffl --test --batch 64
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode term --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --shuffle --percent 0.5 --ratio 1.8 --expname uneven --mode term --test --batch 64

# python train.py --dataset digit --percent 0.5 --expname even --mode fpl --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --expname even --mode fpl --test --batch 64
# python train.py --dataset digit --percent 0.5 --expname even --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --expname even --mode fedsam --test --batch 64

# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedsam --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedsam --test --batch 64


# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode only_dcnet --iters 50 --batch 64 --test_freq 51 --w_con 0.01
# python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode only_dcnet --test --batch 64

# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode ccop --clscon --iters 2 --batch 64 --test_freq 51


# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode ccop --clscon --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode ccop --clscon --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode ccop --clscon --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode ccop --clscon --test --batch 64

# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode only_dcnet --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode only_dcnet --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode only_dcnet --iters 60 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode only_dcnet --test --batch 64

# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51
# python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode ablation --test --batch 64
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode ablation --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.25 --distinct --ratio 1.8 --expname uneven --mode ablation --test --batch 64

python train.py --dataset domainnet --percent 0.5 --distinct --shuffle --ratio 1.6 --expname uneven --mode ccop --clscon --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --distinct --shuffle --ratio 1.6 --expname uneven --mode ccop --clscon --test --batch 64
python train.py --dataset digit --percent 0.25 --distinct --shuffle --ratio 1.8 --expname uneven --mode ccop --clscon --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.25 --distinct --shuffle --ratio 1.8 --expname uneven --mode ccop --clscon --test --batch 64

python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode only_dcnet --iters 50 --batch 64 --test_freq 51 --w_con 0.01
python train.py --dataset domainnet --percent 0.5 --distinct --ratio 1.6 --expname uneven --mode only_dcnet --test --batch 64 --w_con 0.01

python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedavg --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedavg --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode afl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode afl --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode q-ffl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode q-ffl --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode term --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode term --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode harmo-fl --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedmix --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedmix --test --batch 64
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedsam --iters 50 --batch 64 --test_freq 51
python train.py --dataset domainnet --percent 0.1 --distinct --expname even --mode fedsam --test --batch 64

python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedavg --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedavg --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode afl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode afl --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode q-ffl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode q-ffl --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode term --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode term --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode harmo-fl --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode harmo-fl --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedmix --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedmix --test --batch 64
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedsam --iters 50 --batch 64 --test_freq 51
python train.py --dataset digit --percent 0.1 --distinct --expname even --mode fedsam --test --batch 64

