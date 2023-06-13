python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix4 --mode CoCoOP --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix4 --mode CoCoOP --iters 50 --batch 64 --test

python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix5 --mode fedavg --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix5 --mode fedavg --iters 50 --batch 64 --test