python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode ccop --ratio 1.31 \
                --gender_dis random_dis --cluster_num 3 --split_test --test
# python train.py --dataset fairface --percent 1 --expname uneven --binary_race --gender_label --mode ccop --ratio 1.31 \
#                 --gender_dis random_dis --cluster_num 2 --split_test --test
python train.py --dataset fairface --percent 1 --expname uneven --binary_race --mode CoCoOP --ratio 1.31 \
                --gender_dis random_dis --split_test --test
# # # # 4 * (4+2) = 24