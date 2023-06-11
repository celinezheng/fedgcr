# rerun domainnet
# python train.py --dataset domainnet --percent 1 --expname uneven --cb --ratio 1.4 --mode ccop --iters 50 --batch 64 
# python train.py --dataset domainnet --percent 1 --expname uneven --cb --ratio 1.4 --mode ccop --iters 50 --batch 64 --test
# python train.py --dataset domainnet --percent 1 --expname uneven --cb --ratio 1.6 --mode ccop --iters 50 --batch 64 
# python train.py --dataset domainnet --percent 1 --expname uneven --cb --ratio 1.6 --mode ccop --iters 50 --batch 64 --test
# test digit-5 CV
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.5 --mode fedavg --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.5 --mode drfl --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.5 --mode q-ffl --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.5 --mode CoCoOP --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.8 --mode fedavg --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.8 --mode drfl --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.8 --mode q-ffl --iters 50 --test
python train.py --dataset digit --percent 1 --expname uneven --ratio 1.8 --mode CoCoOP --iters 50 --test