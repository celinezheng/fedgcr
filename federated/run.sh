
python train.py --dataset domainnet --percent 0.2 --expname uneven-1-scratch --mode CoCoOP --iters 50 \
    --hparams '{"scratch": "True"}' 
python train.py --dataset domainnet --percent 0.2 --expname uneven-1-scratch --mode CoCoOP --test \
    --hparams '{"scratch": "True"}' 
