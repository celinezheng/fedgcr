# python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode ccop --cluster 7 --quan 0.7 --iters 50 --batch 64
# python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode ccop --cluster 7 --quan 0.7 --iters 50 --batch 64 --test

python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode ccop --cb --cluster 7 --quan 0.7 --iters 50 --batch 64
python train.py --dataset fairface --percent 0.5 --expname even-mix5 --mix5 --mode ccop --cb --cluster 7 --quan 0.7 --iters 50 --batch 64 --test