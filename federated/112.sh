# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode ccop --split_test --ratio 1.31 --iters 50 --cluster_num 2 --quan 0.0 --resume --save_all_gmap --freeze_ckpt --gender_label 
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode ccop --split_test --ratio 1.31 --iters 50 --cluster_num 3 --quan 0.7 --resume --save_all_gmap --freeze_ckpt --gender_label
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode ccop --split_test --ratio 1.31 --iters 50 --cluster_num 4 --quan 0.7 --resume --save_all_gmap --freeze_ckpt --gender_label

# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --iter 50 --gender_label --lr 0.001
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --ratio 1.31 --test --gender_label
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 2 --gender_label
python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 3 --gender_label
python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 4 --gender_label

# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --iter 50
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 2 
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 3 
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode fedavg --split_test --ratio 1.31 --test --cluster_num 4 
