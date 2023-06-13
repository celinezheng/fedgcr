
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix4 --mode fedavg --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix4 --mode fedavg --iters 50 --batch 64 --test

python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix5 --mode CoCoOP --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname uneven --ratio 1.31 --mix5 --mode CoCoOP --iters 50 --batch 64 --test



# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"