from easydict import EasyDict
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

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


class CosineClassificationLoss(torch.nn.modules.loss._Loss):
    def __init__(self, class_num = 10, reduction = 'mean'):
        super(CosineClassificationLoss, self).__init__()
        self.class_num = class_num
        self.reduction = reduction
        self.cosine_similarity = torch.nn.CosineSimilarity()

    def forward(self, pred, target):
        one_hot_target = torch.zeros_like(pred)
        one_hot_target[list(range(pred.size(0))), target] = 1
        minus_cosine_similarity = 1 - self.cosine_similarity(pred, one_hot_target)
        if self.reduction == 'mean':
            loss = torch.mean(minus_cosine_similarity)
        else:
            loss = torch.sum(minus_cosine_similarity)

        return loss

class TrainingConfing(TrainingConfigBase):

    lib_dir = lib_dir

    num_epochs = 180
    val_interval = 10

    create_optimizer = SGDOptimizerMaker(lr =1e-1, momentum = 0.9, weight_decay = 1e-4)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [70, 120, 150], gamma = 0.1)

    create_loss_function = CosineClassificationLoss

    create_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 20, norm = np.inf,
                              mean = torch.tensor(np.array([0.4914, 0.4822, 0.4465]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std = torch.tensor(np.array([0.2023, 0.1994, 0.2010]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0.4914, 0.4822, 0.4465]).astype(np.float32)[np.newaxis, :, np.newaxis,
                                  np.newaxis]),
                              std=torch.tensor(np.array(
                                  [0.2023, 0.1994, 0.2010]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))


config = TrainingConfing()


# About data
# C.inp_chn = 1
# C.num_class = 10

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                 help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                 metavar='N', help='mini-batch size')
parser.add_argument('-d', type=int, default=7, help='Which gpu to use')
parser.add_argument('-freq', '--attack-interval', default=2, type = int,
                    help = 'Specify how many iterations between two batch of adv images')
parser.add_argument('--auto-continue', default=False, action = 'store_true',
                    help = 'Continue from the latest checkpoint')
args = parser.parse_args()


if __name__ == '__main__':
    pass
