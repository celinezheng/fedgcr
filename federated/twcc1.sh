# python train.py --dataset fairface --percent 0.5 --expname even-quan-q0.5-lrc0.01 --quan 0 --mode ccop --iters 50 --cluster 7 --q 0.5\
#     --hparams '{"lr_classifier": 1e-2}'
# python train.py --dataset fairface --percent 0.5 --expname even-quan-q0.5-lrc0.01 --quan 0 --mode ccop --test --cluster 7 --q 0.5

# python train.py --dataset fairface --percent 0.5 --expname even-quan-q0.5-lrm0.01 --quan 0 --mode ccop --iters 50 --cluster 7 --q 0.5\
#     --hparams '{"lr_project": 1e-2}'
# python train.py --dataset fairface --percent 0.5 --expname even-quan-q0.5-lrm0.01 --quan 0 --mode ccop --test --cluster 7 --q 0.5

python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode ccop --iters 50 --quan 0.7 --cluster 7 
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.47 --mode ccop --iters 50 --quan 0.7 --cluster 7 --test

python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode ccop --iters 50 --quan 0.7 --cluster 7 --color_jotter --q 2
python train.py --dataset fairface --percent 1 --expname uneven --ratio 1.31 --mode ccop --iters 50 --quan 0.7 --cluster 7 --color_jotter --q 2 --test

python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --color_jotter --q 2
python train.py --dataset fairface --percent 0.5 --expname even --quan 0.7 --mode ccop --iters 50 --cluster 7 --color_jotter --q 2 --test




# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"