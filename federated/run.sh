# python fed_digits.py --wk_iters 5 --iters 10 --expname 0.01data --mode FedAvg --lr 1e-2
python fed_digits.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 5 --expname 0.01data
python fed_digits.py --hparams '{"lr_classifier": 1e-3}' --wk_iters 10 --iters 5 --expname 0.01data
# python fed_digits.py --expname 0.01data --mode FedAvg --test
python fed_digits.py --lsim --expname 0.01data --test
python fed_digits.py --expname 0.01data --test

# python fed_office.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 5 --expname mse-prompt-out-0.6data
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 5 --expname mse-prompt-out-0.3data
# python fed_digits.py --wk_iters 10 --iters 5 --expname 0.1data --mode FedAvg 
# python fed_office.py --wk_iters 10 --iters 5 --expname 0.6data --mode FedAvg 
# python fed_domainnet.py --wk_iters 5 --iters 10 --expname 0.05datadiff --mode FedAvg --lr 1e-2
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 5 --iters 10 --expname 0.05datadiff
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 5 --iters 10 --expname 0.05datadiff --resume
# python fed_digits.py --expname 0.1data --test --mode FedAvg
# python fed_digits.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --expname mse-prompt-out-0.1data --test


# python fed_office.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 5
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}'  --lsim --wk_iters 10 --iters 3
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}'  --wk_iters 10 --iters 3
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --test --batch 1
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --test --batch 12