# python train.py --dataset fairface --percent 0.5 --expname even --mode q-ffl --iters 50 --gender_dis random_dis
# python train.py --dataset fairface --percent 0.5 --expname even --mode q-ffl --test --gender_dis random_dis
# python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode q-ffl --iters 50 --gender_dis random_dis
# python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode q-ffl --test --gender_dis random_dis
# python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode q-ffl --iters 50 --gender_dis random_dis
# python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode q-ffl --test --gender_dis random_dis

# python train.py --dataset fairface --percent 0.5 --expname even --mode drfl --iters 50 --gender_dis random_dis
# python train.py --dataset fairface --percent 0.5 --expname even --mode drfl --test --gender_dis random_dis
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode drfl --iters 50 --gender_dis random_dis
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode drfl --test --gender_dis random_dis
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode drfl --iters 50 --gender_dis random_dis
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode drfl --test --gender_dis random_dis




# tar -zcvf - ./checkpoint |ssh shulingcheng@140.1150.450.509 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_150/ccop_q=1 |ssh u84505390@5003.145.5016.5030 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"