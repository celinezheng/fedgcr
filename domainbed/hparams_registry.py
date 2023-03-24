import numpy as np
from domainbed import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    
    _hparam('vit_base_16', True, lambda r: True)
    _hparam('im21k', False, lambda r: False)
    _hparam('resnet_dropout', 0.1, lambda r: 0.1)
    _hparam('attention_dropout', 0.0, lambda r: 0.0)

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if "prompt" in algorithm.lower() or 'coop' in algorithm.lower() or algorithm in ["PADA"]:
        _hparam('prompt_dim', 4, lambda r: 4)
        _hparam('lambda', 10, lambda r: 1.0)
        _hparam('lr_prompt', 1e-3, lambda r: 1e-3)
        _hparam('lr_project', 1e-4, lambda r: 1e-4)
        _hparam('wd_project', 1e-5, lambda r: 1e-5)
        _hparam('wd_classifier', 1e-2, lambda r: 1e-2)

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-6, lambda r: 5e-6)
    _hparam('lr_classifier', 5e-4, lambda r: 5e-4)

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        # _hparam('weight_decay', 1e-4, lambda r: 1e-4)
        _hparam('weight_decay', 1e-2, lambda r: 1e-2)

    _hparam('batch_size', 12, lambda r: 12)
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
