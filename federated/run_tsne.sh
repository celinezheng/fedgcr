
# python tsne.py --dataset domainnet --percent 0.1 --expname uneven --mix5 --ratio 1.4 --mode ccop
# python tsne.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop
python tsne_domain.py --dataset officecaltechhome --percent 1 --ratio 1.4 --expname uneven --mode ccop
# python tsne_class.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedavg


# python tsne.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode ccop  --cluster 7
# python tsne.py --dataset fairface --percent 0.5 --expname uneven-mix5 --mix5 --ratio 1.31 --mode fedavg  

# python tsne.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode ccop
# python tsne.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode ccop

# python tsne.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode ccop
# python tsne.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode ccop






# tar -zcvf - ./checkpoint |ssh shulingcheng@140.112.42.29 "tar -zxvf - -C ~/experiment/Fed/FedBN-master/"
# tar -zcvf - ./checkpoint/fed_fairface_uneven-quan_1.47_1_gender_cluster_12/ccop_q=1 |ssh u8425390@203.145.216.230 -p 50558 "tar -zxvf - -C ~/fedprompt/checkpoint"