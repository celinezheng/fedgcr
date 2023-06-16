python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode CoCoOP --iters 50 --batch 64 --split_test 
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode CoCoOP --iters 50 --test --split_test 

python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode q-ffl --iters 50 --test --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode drfl --iters 50 --test --split_test

