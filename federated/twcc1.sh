
# python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode ccop --cs --iters 50 --batch 64 --cluster 4 --split_test
# python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode ccop --cs --iters 50 --test --cluster 4 --split_test

python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode CoCoOP --cluster 4 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode fedavg --cluster 4 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode drfl  --cluster 4 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode q-ffl --cluster 4 --test --batch 64 --split_test

python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode CoCoOP --cluster 5 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode fedavg --cluster 5 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode drfl  --cluster 5 --test --batch 64 --split_test
python train.py --dataset fairface --ratio 2.15 --expname uneven --weak_white --mode q-ffl --cluster 5 --test --batch 64 --split_test



# python train.py --dataset fairface --ratio 1.47 --cs --expname uneven-mix5 --mix5 --mode ccop --iters 50 --batch 64 --cluster 7 --split_test
# python train.py --dataset fairface --ratio 1.47 --cs --expname uneven-mix5 --mix5 --mode ccop --iters 50 --test --cluster 7 --split_test

# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"