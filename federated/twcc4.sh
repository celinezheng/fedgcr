python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode fedavg --iters 50 --batch 64 
python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode fedavg --iters 50 --test --batch 64

python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode drfl --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode drfl --iters 50 --batch 64 --test

