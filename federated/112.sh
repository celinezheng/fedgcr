python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 16 --netdb
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 16 --netdb

python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 16 --netdb
python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 16 --netdb

python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 16 --netdb
python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --cluster 7 --mode ccop --test --batch 16 --netdb

python train.py --dataset fairface --ratio 2.15 --split_test --expname uneven --weak_white --mode ccop --cluser 4 --iters 50 --batch 16 
python train.py --dataset fairface --ratio 2.15 --split_test --expname uneven --weak_white --mode ccop --cluser 4 --iters 50 --test --batch 16

python train.py --dataset fairface --ratio 2.15 --split_test --expname uneven --weak_white --mode ccop --cluser 5 --iters 50 --batch 16 
python train.py --dataset fairface --ratio 2.15 --split_test --expname uneven --weak_white --mode ccop --cluser 5 --iters 50 --test --batch 16
