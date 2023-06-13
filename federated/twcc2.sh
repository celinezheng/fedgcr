# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix --mode CoCoOP --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix --mode CoCoOP --test --batch 64

# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode CoCoOP --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode CoCoOP --test --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode fedavg --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode fedavg --test --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode drfl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode drfl --test --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode ccop --iters 50 --batch 64 --cluster 7
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix2 --mode ccop --test --batch 64 --cluster 7

# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix2 --mode drfl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix2 --mode drfl --test --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix2 --mode q-ffl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix2 --mode q-ffl --test --batch 64


python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix4 --mode fedavg --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix4 --mode fedavg --test --batch 64

python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix5 --mode CoCoOP --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.47 --mix5 --mode CoCoOP --test --batch 64

