
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --iters 50 --cluster_num 13 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --test --cluster_num 13 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --iters 50 --cluster_num 14 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --test --cluster_num 14 --quan 0.7

# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 13 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 13 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 14 --quan 0.7
# python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 14 --quan 0.7


tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"