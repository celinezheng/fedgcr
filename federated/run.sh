# python train.py --dataset office --percent 1 --wk_iters 5 --iters 10 --expname baseline --mode FedPer --lr 1e-2
# python train.py --dataset digit --percent 0.1 --wk_iters 5 --iters 20 --expname 2ClientPerDomain --mode FedPrompt --lr 1e-2
# python train.py --dataset digit --percent 0.1 --wk_iters 5 --iters 20 --expname 2ClientPerDomain --mode DoPrompt --lr 1e-2 --resume
# python train.py --dataset digit --percent 0.1 --wk_iters 1 --iters 1 --expname uneven --mode FedPrompt --lr 1e-2
# python train.py --dataset digit --percent 0.1 --wk_iters 5 --iters 10 --expname uneven --mode FedPrompt --lr 1e-2
# python train.py --dataset digit --percent 0.1 --wk_iters 5 --iters 10 --expname uneven --mode DoPrompt --lr 1e-2
# python train.py --dataset digit --percent 0.1 --expname uneven --mode DoPrompt --lr 1e-2 --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 40 --expname uneven --mode DoPrompt --lr 1e-2 --resume
# python train.py --dataset digit --percent 0.1 --wk_iters 5 --iters 10 --expname uneven_alignonce --mode FedPrompt --lr 1e-2
# python train.py --dataset digit --percent 0.1 --expname uneven_alignonce --mode FedPrompt --lr 1e-2 --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname uneven_alignonce --mode FedPrompt --lr 1e-2 --batch 16
# python train.py --dataset domainnet --percent 0.05 --expname uneven_alignonce --mode FedPrompt --lr 1e-2 --test --batch 8



python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode DoPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-1-sqrt --mode DoPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode FedPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode FedPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode PromptFL --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-1 --mode PromptFL --lr 1e-2 --batch 8 --test 


python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode FedPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode DoPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode DoPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode CoCoOP --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-2-sqrt --mode CoCoOP --lr 1e-2 --batch 8 --test 
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode FedPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode DoPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode DoPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode CoCoOP --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset domainnet --percent 0.05 --expname uneven-2-sqrt --mode CoCoOP --lr 1e-2 --batch 8 --test 
python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode FedPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset digit --percent 0.1 --expname uneven-2 --mode FedPrompt --lr 1e-2 --batch 8 --test 
python train.py --dataset domainnet --percent 0.05 --expname uneven-2 --mode FedPrompt --lr 1e-2 --batch 16 --wk_iters 5 --iters 10
python train.py --dataset domainnet --percent 0.05 --expname uneven-2 --mode FedPrompt --lr 1e-2 --batch 8 --test 



### dg
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Clipart --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Clipart --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Clipart --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Clipart --mode DoPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Infograph --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Infograph --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Infograph --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Infograph --mode DoPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Painting --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Painting --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Painting --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Painting --mode DoPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain QuickDraw --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain QuickDraw --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain QuickDraw --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain QuickDraw --mode DoPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Real --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Real --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Real --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Real --mode DoPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Sketch --mode FedPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --wk_iters 5 --iters 10 --expname dg_uneven --target_domain Sketch --mode DoPrompt --lr 1e-2
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Sketch --mode FedPrompt --test --batch 8
# python train.py --dataset domainnet --percent 0.05 --expname dg --target_domain Sketch --mode DoPrompt --test --batch 8
