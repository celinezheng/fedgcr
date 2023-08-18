python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedavg --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedavg --test

python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode afl --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode afl --test
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode q-ffl --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode q-ffl --test
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode term --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode term --test

python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode harmo-fl --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode harmo-fl --test
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedsam --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedsam --test
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedmix --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode fedmix --test

python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode ccop --iters 50
python train.py --dataset pacs --percent 1 --seed 1 --expname uneven --ratio 2.16 --shuffle --mode ccop --test
