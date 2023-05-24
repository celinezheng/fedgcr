
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --cluster_num 9
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --cluster_num 9

# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --gender_dis single --cluster_num 9
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --gender_dis single --cluster_num 9
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --gender_dis single --cluster_num 7
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --gender_dis single --cluster_num 7

# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --cluster_num 9
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --cluster_num 9
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --iters 50 --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --test --gender_dis random_dis --cluster_num 7

python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --iters 50 --gender_dis random_dis --cluster_num 7
python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --test --gender_dis random_dis --cluster_num 7


python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --sam
python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --sam



# # # # 4 * (4+2) = 24