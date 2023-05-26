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

python train.py --dataset fairface --percent 0.5 --expname uneven-stdrw --mode ccop --iters 30 --gender_dis gender --cluster_num 7 --batch 16 --std_rw --small_test --ratio 1.7
python train.py --dataset fairface --percent 0.5 --expname uneven-stdrw --mode ccop --test --gender_dis gender --cluster_num 7 --std_rw --small_test --ratio 1.7


# python train.py --dataset fairface --percent 0.2 --expname even-stdrw --mode ccop --iters 5 --gender_dis random_dis --cluster_num 7  --batch 16
# python train.py --dataset fairface --percent 0.2 --expname even-stdrw --mode ccop --test --gender_dis random_dis --cluster_num 7

# # # # 4 * (4+2) = 24