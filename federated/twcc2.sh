python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 2
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 2

python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 5
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 5

python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --iters 50 --batch 64 --cs --power_cs 10
python train.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --cluster 7 --mode ccop --test --batch 64 --cs --power_cs 10