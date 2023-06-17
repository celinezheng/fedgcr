python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 2
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 2

# run at 112
# python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 5
# python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 5
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 1 --super_quan
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 1 --super_quan

