python train.py --dataset digit --seed 1 --expname even --split_test --mode fedavg --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode afl --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode q-ffl --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode term --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode harmo-fl --test


python train.py --dataset digit --seed 1 --expname even --split_test --mode fedsam --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode fedmix --test
python train.py --dataset digit --seed 1 --expname even --split_test --mode ccop --test

