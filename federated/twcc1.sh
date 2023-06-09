
# python train.py --dataset fairface --percent 1 --expname uneven --quan 0.7 --ratio 1.31 --mode ccop --iters 50 --cluster 7 --batch 64
# python train.py --dataset fairface --percent 1 --expname uneven --quan 0.7 --ratio 1.31 --mode ccop --iters 50 --cluster 7 --batch 64 --test
# python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --batch 64 --test

python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode CoCoOP --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode CoCoOP --iters 50 --cluster 7 --batch 64 --test
python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --iters 50 --cluster 7 --batch 64 --test

python train.py --dataset fairface --percent 1 --expname uneven --cb --ratio 1.31 --mode ccop --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 1 --expname uneven --cb --ratio 1.31 --mode ccop --iters 50 --cluster 7 --batch 64 --test
python train.py --dataset fairface --percent 0.5 --expname even --cb --mode ccop --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --cb --mode ccop --iters 50 --cluster 7 --batch 64 --test

python train.py --dataset fairface --percent 0.5 --expname even --mode drfl --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --mode drfl --iters 50 --batch 64 --test
python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --iters 50 --batch 64 --test
python train.py --dataset fairface --percent 0.5 --expname even --mode q-ffl --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --mode q-ffl --iters 50 --batch 64 --test
python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --batch 64 --test

python train.py --dataset fairface --percent 1 --expname uneven --cb --ratio 1.47 --mode ccop --iters 50 --cluster 7 --batch 64
python train.py --dataset fairface --percent 1 --expname uneven --cb --ratio 1.47 --mode ccop --iters 50 --cluster 7 --batch 64 --test
# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"