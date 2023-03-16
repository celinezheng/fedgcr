python fed_digits.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 5
python fed_digits.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 5 
python fed_digits.py --hparams '{"lr_classifier": 1e-3}' --lsim --test
python fed_digits.py --hparams '{"lr": 1e-5, "lr_classifier": 1e-3}' --test
# python fed_office.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 5 
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 0
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 2
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --wk_iters 10 --iters 3 --target_idx 3
# python fed_office.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 5
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 3 --target_idx 2
# python fed_office_dg.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --wk_iters 10 --iters 3 --target_idx 3
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}'  --lsim --wk_iters 10 --iters 3
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}'  --wk_iters 10 --iters 3
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --lsim --test --batch 1
# python fed_domainnet.py --hparams '{"lr_classifier": 1e-3, "nonlinear_classifier": "True"}' --test --batch 1

# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --wk_iters 10 --iters 3 --target_idx 0
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 3 --target_idx 1 --resume

# python fed_digits_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 3 --target_idx 4
# python fed_digits_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 3 --target_idx 4

# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 3 --target_idx 0
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 3 --target_idx 1
python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 3 --target_idx 2 --resume
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 5 --target_idx 2
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 5 --target_idx 3
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 5 --target_idx 3
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 5 --target_idx 4
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 5 --target_idx 4
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}' --lsim --wk_iters 10 --iters 5 --target_idx 5
# python fed_domainnet_dg.py --hparams '{"lr_classifier": 1e-3}'  --wk_iters 10 --iters 5 --target_idx 5
