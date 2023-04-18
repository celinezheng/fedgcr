
# python train.py --dataset digit --percent 0.1 --expname even --mode CoCoOP --iters 30 --wk_iters 2
# python train.py --dataset digit --percent 0.1 --expname even --mode CoCoOP --test --batch 8


# python train.py --dataset domainnet --percent 0.2 --expname even --mode CoCoOP --iters 30 --wk_iters 2
# python train.py --dataset domainnet --percent 0.2 --expname even --mode CoCoOP --test --batch 8
# python train.py --dataset domainnet --percent 0.2 --expname even --mode Nova --iters 30 --wk_iters 2
# python train.py --dataset domainnet --percent 0.2 --expname even --mode Nova --test --batch 8

# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode Nova --iters 30 --wk_iters 2
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode Nova --test --batch 1
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode Nova --iters 30 --wk_iters 2

python train.py --dataset digit --percent 0.1 --expname even --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset digit --percent 0.1 --expname even --mode drfl --test --batch 1 
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode drfl --test --batch 1 
python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode drfl --test --batch 1 

python train.py --dataset domainnet --percent 0.2 --expname even --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset domainnet --percent 0.2 --expname even --mode drfl --test --batch 1 
python train.py --dataset domainnet --percent 0.2 --expname uneven-1 --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset domainnet --percent 0.2 --expname uneven-1 --mode drfl --test --batch 1 
python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode drfl --iters 50 --wk_iters 1 
python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode drfl --test --batch 1 

# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode Nova --iters 50 --wk_iters 1
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode Nova --test --batch 1
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode CoCoOP --iters 50 --wk_iters 1
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode CoCoOP --test --batch 1

# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode Nova --iters 50 --wk_iters 1
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode Nova --test --batch 1
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode CoCoOP --iters 50 --wk_iters 1
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode CoCoOP --test --batch 1
