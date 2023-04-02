
# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt --iters 40
# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt --iters 40
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --iters 40
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8

python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt --iters 30 --wk_iters 5
# python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt --test --batch 8
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt --iters 40
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --iters 40
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8


# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt --test --batch 8 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8 --seed 2

# python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt --test --batch 8 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8 --seed 2

# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode CoCoOP --test --batch 8 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode CoCoOP --test --batch 8 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode CoCoOP --test --batch 8 --seed 2

# python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode CoCoOP --test --batch 8 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode CoCoOP --test --batch 8 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode CoCoOP --iters 40 --seed 2
# python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode CoCoOP --test --batch 8 --seed 2