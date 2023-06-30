python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset digit --percent 0.5 --ratio 1.8 --expname uneven --mode fedsam --test --batch 32
python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset digit --percent 0.5 --ratio 1.5 --expname uneven --mode fedsam --test --batch 32
python train.py --dataset digit --percent 0.5 --expname even --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset digit --percent 0.5 --expname even --mode fedsam --test --batch 32
python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset domainnet --percent 1 --ratio 1.4 --expname uneven --mode fedsam --test --batch 32
python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset domainnet --percent 1 --ratio 1.6 --expname uneven --mode fedsam --test --batch 32
python train.py --dataset domainnet --percent 0.5 --expname even --mode fedsam --iters 50 --batch 32 --test_freq 51
python train.py --dataset domainnet --percent 0.5 --expname even --mode fedsam --test --batch 32