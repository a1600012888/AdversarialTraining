# AdversarialTraining

This is an inplementaion of Adversarial Training on Python3, Pytorch. The code is designed to be easy to be extended.

This repository now includes:
* Natural training code for CIFAR-10 (see experiments/CIFAR10/adversaril.train)
* Adversaril Training code for CIFAR-10 (see experiments/CIFAR10/adversaril.train)
* PGD attack with different norm constraint, e.g. L-1, L2, L-$\inf$ (see lib/attack/pgd.py)
* TRADES attack and TRADES training. (which won the 1st place in the robust model track NeurIPS 2018 Adversarial Vision Challenge)

This repository will includes:
* Examples of training on your own dataset, using your own models, your own loss functions or training againt your own adversaries
* More popular attack methods
* More base models

