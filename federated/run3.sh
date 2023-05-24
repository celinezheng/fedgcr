# python train.py --dataset fairface --percent 0.5 --expname uneven-debug --mode ccop --ratio 1.7 --iters 3 --gender_dis gender --cluster_num 11 --small_test --debug
# python train.py --dataset fairface --percent 0.5 --expname uneven --mode ccop --ratio 1.7 --test --gender_dis gender --cluster_num 11 --small_test 

# python train.py --dataset fairface --percent 0.5 --expname uneven --mode CoCoOP --ratio 1.7 --iters 30 --gender_dis gender --small_test
# python train.py --dataset fairface --percent 0.5 --expname uneven --mode CoCoOP --ratio 1.7 --test --gender_dis gender --small_test 

# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --gender_dis gender --cluster_num 12 --debug
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --gender_dis gender --cluster_num 12
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --gender_dis gender --cluster_num 13
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --gender_dis gender --cluster_num 13
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --iters 50 --gender_dis gender 
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --test --gender_dis gender 
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50 --gender_dis gender --cluster_num 14
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --gender_dis gender --cluster_num 14

# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 50
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --iters 50
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.31 --test

# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --iters 50
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --test

python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --iters 50
python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --test
python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --iters 50
python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --test
python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --iters 50
python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --test

# # # # 4 * (4+2) = 24