
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --iters 50 --cluster_num 14 --quan 0.5
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --test --cluster_num 14 --quan 0.5

# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 13 --quan 0.5
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 13 --quan 0.5
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 14 --quan 0.5
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 14 --quan 0.5


python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --cluster 7 --quan 0.5 --iters 50
python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --cluster 7 --quan 0.5 --test

python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --gender_dis gender --iters 50
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --gender_dis gender --test
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --gender_dis gender --iters 50
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --gender_dis gender --test





# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"