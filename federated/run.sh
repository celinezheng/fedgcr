
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 13
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 14
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --gender_dis gender --test
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --gender_dis gender --test
