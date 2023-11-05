python train.py --dataset domainnet --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.3 --iters 50
python train.py --dataset domainnet --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.3 --test
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.4 --iters 50
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.4 --test
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.6 --iters 50
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.6 --test

python train.py --dataset digit --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.3 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.3 --test
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.5 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --beta 0.3 --ratio 1.5 --test
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --bat 0.3 --ratio 1.8 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --bat 0.3 --ratio 1.8

python train.py --dataset domainnet --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.1 --iters 50
python train.py --dataset domainnet --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.1 --test
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.4 --iters 50
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.4 --test
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.6 --iters 50
python train.py --dataset domainnet --percent 1 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.6 --test

python train.py --dataset digit --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.1 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname even --mode fedgcr --beta 0.1 --test
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.5 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --beta 0.1 --ratio 1.5 --test
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --bat 0.1 --ratio 1.8 --iters 50
python train.py --dataset digit --percent 0.5 --seed 1 --expname uneven --mode fedgcr --bat 0.1 --ratio 1.8