# python train.py --dataset domainnet --percent 1 --seed 0 --expname uneven --ratio 1.6 --cluster kmeans --mode ccop --iters 2


python train.py --dataset domainnet --percent 1 --seed 0 --expname uneven --ratio 1.6 --cluster kmeans --mode ccop --iters 50
python train.py --dataset domainnet --percent 1 --seed 0 --expname uneven --ratio 1.6 --cluster kmeans --mode ccop --test
python train.py --dataset domainnet --percent 1 --seed 2 --expname uneven --ratio 1.6 --cluster kmeans --mode ccop --iters 50
python train.py --dataset domainnet --percent 1 --seed 2 --expname uneven --ratio 1.6 --cluster kmeans --mode ccop --test
