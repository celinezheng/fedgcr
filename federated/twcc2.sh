

# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode fedavg --iters 50 --batch 64 --split_test 
# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode fedavg --iters 50 --test --split_test 

# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode drfl --iters 50 --batch 64 --split_test
# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode drfl --iters 50 --test --split_test

# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --cs --iters 50 --batch 64 --cluster 3 --split_test
# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --cs --iters 50 --test --cluster 3 --split_test

# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --quan 0.7 --iters 50 --batch 64 --cluster 3 --split_test
# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --quan 0.7 --iters 50 --test --cluster 3 --split_test


# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --cs --q 0 --iters 70 --test_freq 5 --save_mean 0.45 --batch 64 --cluster 3 --split_test
# python train.py --dataset fairface --ratio 1.7 --expname uneven --weak_white --mode ccop --cs --q 0 --iters 70 --test_freq 5 --save_mean 0.45 --test --cluster 3 --split_test

python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode fedavg --iters 50 --batch 64 --split_test 
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode fedavg --iters 50 --test --split_test 

python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode ccop --cs --iters 50 --batch 64 --cluster 5 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode ccop --cs --iters 50 --test --cluster 5 --split_test

python train.py --dataset fairface --ratio 1.47 --expname uneven-mix5 --mix5 --mode fedavg --iters 50 --batch 64 --split_test
python train.py --dataset fairface --ratio 1.47 --expname uneven-mix5 --mix5 --mode fedavg --iters 50 --test --split_test

python train.py --dataset fairface --ratio 1.47 --expname uneven-mix5 --mix5 --mode drfl --iters 50 --batch 64 --split_test
python train.py --dataset fairface --ratio 1.47 --expname uneven-mix5 --mix5 --mode drfl --iters 50 --test --split_test

