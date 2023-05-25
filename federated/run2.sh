

python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --gender_dis random_dis --cluster_num 7 --color_jitter
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --gender_dis random_dis --cluster_num 7 --color_jitter

python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --iters 50 --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --test --gender_dis random_dis --cluster_num 7

python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --sam



# # # 4 * (4+2) = 24