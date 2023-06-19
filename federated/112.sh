python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --mode fedavg --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --mode fedavg --test --batch 16 --netdb

python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --mode drfl --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.47 --expname uneven-mix5 --mix5 --mode drfl --test --batch 16 --netdb

python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --mode fedavg --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --mode fedavg --test --batch 16 --netdb

python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --mode drfl --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 1.31 --expname uneven-mix5 --mix5 --mode drfl --test --batch 16 --netdb

python change-vpt.py --dataset fairface --percent 0.5 --ratio 2.15 --expname uneven --weak_whte --split_test --mode fedavg --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 2.15 --expname uneven --weak_whte --split_test --mode fedavg --test --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 2.15 --expname uneven --weak_whte --split_test --mode drfl --iters 50 --batch 16 --netdb
python change-vpt.py --dataset fairface --percent 0.5 --ratio 2.15 --expname uneven --weak_whte --split_test --mode drfl --test --batch 16 --netdb


