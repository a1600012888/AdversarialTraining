'''
Reference:
[1] Towards Deep Learning Models Resistant to Adversarial Attacks
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
arXiv:1706.06083v3
'''
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
if not father_dir in sys.path:
    sys.path.append(father_dir)
from attack.attack_base import AttackBase, clip_eta

class IPGDTrades(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 8 / 255.0, sigma = 2 / 255.0, nb_iter = 20,
                 norm = np.inf, DEVICE = torch.device('cpu'),
                 mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.KLDivLoss(size_average=False).to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)

    def single_attack(self, net, inp, pred_natural, eta):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param pred_natural:  predictions on
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''

        adv_inp = inp + eta

        net.zero_grad()

        pred = net(adv_inp)

        loss = self.criterion(F.log_softmax(pred, dim = 1),F.softmax(pred_natural, dim=1))
        grad_sign = torch.autograd.grad(loss, adv_inp,
                                        only_inputs=True, retain_graph = False)[0].sign()

        adv_inp = adv_inp + grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean

        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, *inputs):

        eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        with torch.no_grad():
            pred_natural = net(inp)
        for i in range(self.nb_iter):
            eta = self.single_attck(net, inp, pred_natural, eta)
            #print(i)

        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

def test_IPGDTrades():
    pass
if __name__ == '__main__':
    test_IPGDTrades()
