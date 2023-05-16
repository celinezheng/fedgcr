# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test 
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --iters 50 
# python train.py --dataset fairface --percent 1 --expname uneven --mode CoCoOP --ratio 1.47 --test 

# python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --iters 50
# python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --test 
# python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --iters 50 
# python train.py --dataset fairface --percent 0.5 --expname even --mode fedavg --test 
# python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --iters 50 
# python train.py --dataset fairface --percent 0.5 --expname even --mode CoCoOP --test 
# python train.py --dataset fairface --percent 0.5 --expname even --mode harmo-fl --iters 50 
# python train.py --dataset fairface --percent 0.5 --expname even --mode harmo-fl --test 


python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --iters 1
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.31 --test --sam
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --iters 50 --sam
# python train.py --dataset fairface --percent 1 --expname uneven --mode ccop --ratio 1.47 --test --sam

# python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --iters 50 
# python train.py --dataset fairface --percent 1 --expname uneven --mode fedavg --ratio 1.31 --test 
# python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --iters 50 --sam
# python train.py --dataset fairface --percent 0.5 --expname even --mode ccop --test --sam


# # # # 4 * (4+2) = 24