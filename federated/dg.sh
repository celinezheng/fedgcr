# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 0
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 2
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 3
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 3 --target_idx 2
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 3 --target_idx 3

# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --wk_iters 10 --iters 3 --target_idx 0
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 3 --target_idx 1 --resume

# python fed_digits_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 3 --target_idx 4
# python fed_digits_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 3 --target_idx 4

python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 5 --iters 10 --target_idx 0
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 5 --iters 10 --target_idx 0 
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 5 --iters 10 --target_idx 0 --resume
# python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 10 --iters 5 --target_idx 1
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 10 --iters 5 --target_idx 1
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 10 --iters 5 --target_idx 1
# python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 10 --iters 5 --target_idx 2
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 10 --iters 5 --target_idx 2
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 10 --iters 5 --target_idx 2
# python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 10 --iters 5 --target_idx 3
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 10 --iters 5 --target_idx 3
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 10 --iters 5 --target_idx 3
# python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 10 --iters 5 --target_idx 4
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 10 --iters 5 --target_idx 4
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 10 --iters 5 --target_idx 4
# python fed_domainnet_dg.py --expname dg0.1data --mode FedAvg --wk_iters 10 --iters 5 --target_idx 5
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --lsim --wk_iters 10 --iters 5 --target_idx 5
# python fed_domainnet_dg.py --expname dg0.1data --hparams '{"lr_classifier": 1e-2, "lr_prompt": 1e-2}' --wk_iters 10 --iters 5 --target_idx 5