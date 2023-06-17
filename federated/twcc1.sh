


python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --iters 50 --batch 64 --cs --power_cs 2
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --test --batch 64 --cs --power_cs 2

python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --iters 50 --batch 64 --cs --power_cs 5
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --test --batch 64 --cs --power_cs 5

python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --iters 50 --batch 64 --cs --power_cs 10
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ablation --test --batch 64 --cs --power_cs 10



# python train.py --dataset digit --ratio 1.5 --expname uneven --cluster 5 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset digit --ratio 1.5 --expname uneven --cluster 5 --mode ccop --test --batch 64 --cs
# python train.py --dataset digit --ratio 1.8 --expname uneven --cluster 5 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset digit --ratio 1.8 --expname uneven --cluster 5 --mode ccop --test --batch 64 --cs

# python train.py --dataset domainnet --ratio 1.4 --expname uneven --cluster 6 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset domainnet --ratio 1.4 --expname uneven --cluster 6 --mode ccop --test --batch 64 --cs
# python train.py --dataset domainnet --ratio 1.6 --expname uneven --cluster 6 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset domainnet --ratio 1.6 --expname uneven --cluster 6 --mode ccop --test --batch 64 --cs

# python train.py --dataset digit --expname even --cluster 5 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset digit --expname even --cluster 5 --mode ccop --test --batch 64 --cs
# python train.py --dataset domainnet --expname even --cluster 6 --mode ccop --iters 50 --batch 64 --cs
# python train.py --dataset domainnet --expname even --cluster 6 --mode ccop --test --batch 64 --cs


# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"