from easydict import EasyDict
import sys
import os
import argparse
import numpy as np
import torch

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

abs_current_path = os.path.realpath('./')
root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join(root_path, 'lib')
add_path(lib_dir)

from training.config import TrainingConfigBase, SGDOptimizerMaker, \
    PieceWiseConstantLrSchedulerMaker, IPGDAttackMethodMaker

class IPGDTradesMaker(object):
    def __init__(self, eps, sigma, nb_iters, norm, mean, std):
        self.eps = eps
        self.sigma = sigma
        self.nb_iters = nb_iters
        self.norm = norm
        self.mean = mean
        self.std = std

    def __call__(self, DEVICE):

        from attack.pgd_trades import IPGDTrades
        return IPGDTrades(self.eps, self.sigma, self.nb_iters, self.norm, DEVICE, self.mean, self.std)
class TrainingConfing(TrainingConfigBase):

    lib_dir = lib_dir

    num_epochs = 110
    val_interval = 10
    alpha = 1.0

    create_optimizer = SGDOptimizerMaker(lr =1e-1, momentum = 0.9, weight_decay = 2e-4)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [70, 90, 100], gamma = 0.1)

    create_loss_function = torch.nn.CrossEntropyLoss

    create_attack_method = \
        IPGDTradesMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 10, norm = np.inf,
                              mean = torch.tensor(np.array([0, 0, 0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std = torch.tensor(np.array([1,1,1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0, 0, 0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(
                                  np.array([1, 1, 1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

config = TrainingConfing()


# About data
# C.inp_chn = 1
# C.num_class = 10

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                 help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                 metavar='N', help='mini-batch size')
parser.add_argument('-d', type=int, default=2, help='Which gpu to use')
parser.add_argument('-freq', '--attack-interval', default=2, type = int,
                    help = 'Specify how many iterations between two batch of adv images')
parser.add_argument('--auto-continue', default=False, action = 'store_true',
                    help = 'Continue from the latest checkpoint')
args = parser.parse_args()


if __name__ == '__main__':
    pass
