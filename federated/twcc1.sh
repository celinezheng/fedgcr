# python train.py --dataset digit --percent 1 --seed 0 --expname uneven --ratio 1.4 --mode ccop --cluster_num 4 --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 1 --seed 0 --expname uneven --ratio 1.4 --mode ccop --cluster_num 4 --test --batch 64 --test_freq 51

# python train.py --dataset digit --percent 0.5 --seed 0 --expname uneven --ratio 1.8 --mode ccop --cluster_num 3 --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --seed 0 --expname uneven --ratio 1.8 --mode ccop --cluster_num 3 --test --batch 64 --test_freq 51

# python train.py --dataset digit --percent 0.5 --seed 2 --expname uneven --ratio 1.8 --mode ccop --cluster_num 3 --iters 50 --batch 64 --test_freq 51
# python train.py --dataset digit --percent 0.5 --seed 2 --expname uneven --ratio 1.8 --mode ccop --cluster_num 3 --test --batch 64 --test_freq 51


# python train.py --dataset digit --seed 1 --expname even --split_test --clister_num 3 --mode ccop --test
# python train.py --dataset digit --seed 1 --expname even --split_test --clister_num 7 --mode ccop --test
# python train.py --dataset digit --seed 1 --expname even --split_test --clister_num 4 --mode ccop --test
# python train.py --dataset digit --seed 1 --expname even --split_test --clister_num 6 --mode ccop --test


python train.py --dataset domainnet --percent 1 --seed 1 --expname debug-uneven --ratio 1.4 --mode ccop --iters 2
python train.py --dataset domainnet --percent 1 --seed 1 --expname debug-uneven --ratio 1.4 --mode ccop --test
