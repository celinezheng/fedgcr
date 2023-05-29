

python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --gender_dis random_dis --iters 50
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.47 --gender_dis random_dis --test 
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --gender_dis random_dis --iters 50 
python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --gender_dis random_dis --test 
python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --iters 50 --cluster_num 7 --quan 0.7 
python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --test --cluster_num 7 --quan 0.7 
python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --iters 50 --cluster_num 7 --quan 0.6
python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --test --cluster_num 7 --quan 0.6

# python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --iters 60 --cluster_num 7 --quan 0.6 --std_rw
# python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --test --cluster_num 7 --quan 0.6 --std_rw

# python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --iters 60 --cluster_num 7 --quan 0.7 --std_rw
# python train.py --dataset fairface --percent 1 --expname uneven-quan-std --mode ccop --ratio 1.47 --gender_dis random_dis --test --cluster_num 7 --quan 0.7 --std_rw
