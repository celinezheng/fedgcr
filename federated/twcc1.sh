python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.5
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12 --quan 0.5

python train.py --dataset fairface --percent 1 --expname uneven-quan --mode CoCoOP --ratio 1.47 --gender_dis gender --iters 50 
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode CoCoOP --ratio 1.47 --gender_dis gender --test 
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.5 
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --test --cluster_num 12 --quan 0.5

python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.5 --std_rw
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12 --quan 0.5 --std_rw
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.5 --std_rw 
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.47 --gender_dis gender --test --cluster_num 12 --quan 0.5 --std_rw

python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.6
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12 --quan 0.6
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --iters 50 --cluster_num 12 --quan 0.7
python train.py --dataset fairface --percent 1 --expname uneven-quan --mode ccop --ratio 1.31 --gender_dis gender --test --cluster_num 12 --quan 0.7

