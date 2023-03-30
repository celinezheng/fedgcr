
python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt
python train.py --dataset domainnet --percent 0.05 --expname even-sqrt --mode FedPrompt --test --batch 8
python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt
python train.py --dataset domainnet --percent 0.05 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8

python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt
python train.py --dataset digit --percent 0.1 --expname even-sqrt --mode FedPrompt --test --batch 8
python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt
python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode FedPrompt --test --batch 8
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --test --batch 8