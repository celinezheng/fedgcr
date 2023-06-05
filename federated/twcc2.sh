

# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --iters 50
# python train.py --dataset domainnet --percent 0.5 --expname even --mode fedavg --test
# python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode fedavg --iters 50
# python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode fedavg --test

python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --iters 50
python train.py --dataset domainnet --percent 0.5 --expname even --mode q-ffl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode q-ffl --iters 50
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.4 --mode q-ffl --test
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode q-ffl --iters 50
python train.py --dataset domainnet --percent 1 --expname uneven --ratio 1.6 --mode q-ffl --test



# 5 min 50 epoch, 1 run=1505min, 1test=50min, 1 run=1507 min,
# total =1507*8min = 16hr


# tar -zcvf - ./checkpoint |ssh shulingcheng@140.1150.450.509 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_150/ccop_q=1 |ssh u84505390@5003.145.5016.5030 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"