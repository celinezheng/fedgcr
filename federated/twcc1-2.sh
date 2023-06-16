
# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode fedavg --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode fedavg --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode CoCoOP --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode CoCoOP --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode CoCoOP --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode drfl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode drfl --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode q-ffl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode q-ffl --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode ccop --quan 0.7 --cluster 7 --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode ccop --quan 0.7 --cluster 7 --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode drfl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode drfl --iters 50 --batch 64 --test

# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode q-ffl --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.25 --expname even-mix5 --mix5 --mode q-ffl --iters 50 --batch 64 --test



# python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven --cb --weak_white --mode ccop --cluster 6 --iters 15 --batch 64 
# python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven --cb --weak_white --mode ccop --cluster 6 --iters 50 --batch 64 
# python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven --cb --weak_white --mode ccop --cluster 6 --iters 50 --test --batch 64

# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --quan 0.7 --weak_white --mode ccop --cluster 7 --iters 50 --batch 64 
# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --quan 0.7 --weak_white --mode ccop --cluster 7 --iters 50 --test --batch 64
# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode fedavg --iters 50 --batch 64 
# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode fedavg --iters 50 --test --batch 64
# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode CoCoOP --iters 50 --batch 64 
# python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --weak_white --mode CoCoOP --iters 50 --test --batch 64

python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --quan 0.7 --cb --weak_white --mode ccop --cluster 7 --iters 50 --batch 64 
python train.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven --quan 0.7 --cb --weak_white --mode ccop --cluster 7 --iters 50 --test --batch 64

# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"