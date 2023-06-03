

python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --iters 2
python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode fedavg --iters 2
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode fedavg --test

python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --iters 2
python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode q-ffl --iters 2
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode q-ffl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode q-ffl --iters 2
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode q-ffl --test

python train.py --dataset domainnet --percent 0.5 --expname even --mode drfl --iters 2
python train.py --dataset domainnet --percent 0.5 --expname even --mode drfl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode drfl --iters 2
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode drfl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode drfl --iters 2
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode drfl --test





# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 2558 "tar -zxvf - -C ~/fedprompt/checkpoint"