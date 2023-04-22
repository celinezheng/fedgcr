
# python train.py --dataset digit --percent 0.1 --expname even --mode CoCoOP --iters 30 --wk_iters 2
# python train.py --dataset digit --percent 0.1 --expname even --mode CoCoOP --test --batch 8


# python train.py --dataset domainnet --percent 0.2 --expname even --mode CoCoOP --iters 30 --wk_iters 2
# python train.py --dataset domainnet --percent 0.2 --expname even --mode CoCoOP --test --batch 8
# python train.py --dataset domainnet --percent 0.2 --expname even --mode Nova --iters 30 --wk_iters 2

# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode ccop --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode ccop --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname even --mode full --iters 50 --wk_iters 1 --batch 8
# python train.py --dataset domainnet --percent 0.2 --expname even --mode full --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode drfl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode drfl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode q-ffl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode q-ffl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode fedavg --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode fedavg --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode full --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode full --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode drfl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode drfl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode q-ffl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode q-ffl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode fedavg --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode fedavg --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --iters 50 --wk_iters 1 --si 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --iters 50 --wk_iters 1 --si 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode drfl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode drfl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode q-ffl --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode q-ffl --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode fedavg --iters 50 --wk_iters 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode fedavg --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode ccop --iters 50 --wk_iters 1 --si 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode ccop --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --iters 50 --wk_iters 1 --si 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode ccop --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode ccop --iters 50 --wk_iters 1 --si 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode ccop --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname even --mode ccop --iters 50 --wk_iters 1 --si --q 0
# python train.py --dataset domainnet --percent 0.2 --expname even --mode ccop --test --batch 1 --q 0

# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode drfl --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode drfl --test --batch 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode q-ffl --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode q-ffl --test --batch 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode fedavg --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode fedavg --test --batch 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode ccop --iters 50 --wk_iters 1 --q 0 --si
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode ccop --test --batch 1 --q 0

python train.py --dataset digit --percent 0.1 --expname even --mode solo --iters 50 --wk_iters 1 
python train.py --dataset digit --percent 0.1 --expname even --mode solo --test --batch 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode solo --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode solo --test --batch 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode solo --iters 50 --wk_iters 1 
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode solo --test --batch 1 

# python train.py --dataset domainnet --percent 0.2 --expname even --mode solo --iters 50 --wk_iters 1
# python train.py --dataset domainnet --percent 0.2 --expname even --mode solo --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode solo --iters 50 --wk_iters 1
# python train.py --dataset domainnet --percent 0.2 --expname uneven-4 --mode solo --test --batch 1 
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode solo --iters 50 --wk_iters 1
# python train.py --dataset domainnet --percent 0.2 --expname uneven-2 --mode solo --test --batch 1 

# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode ccop --iters 50 --wk_iters 1 --si
# python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode ccop --test --batch 1 

