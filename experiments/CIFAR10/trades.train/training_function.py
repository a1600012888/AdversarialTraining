import os
import sys
import config
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm
import torch.nn.functional as F


def train_one_epoch(net, batch_generator, optimizer,
                    criterion, DEVICE=torch.device('cuda:0'),
                    descrip_str='Training', AttackMethod = None, alpha = 1):
    '''

    :param AttackMethod: the attack method, None represents natural training
    :param alpha: weight coeffcient for mig loss
    :return:  None    #(clean_acc, adv_acc)
    '''

    #assert callable(AttackMethod)
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    criterion_kl = torch.nn.KLDivLoss(size_average=False).to(DEVICE)
    pbar.set_description(descrip_str)

    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()

        adv_inp = AttackMethod.attack(net, data, label)

        optimizer.zero_grad()
        pred1 = net(adv_inp)
        pred2 = net(data)
        loss_robust = criterion_kl(F.log_softmax(pred1, dim=1), F.softmax(pred2, dim = 1))
        loss_natural = criterion(pred2, label)
        TotalLoss = loss_natural + alpha * loss_robust

        TotalLoss.backward()

        acc = torch_accuracy(pred1, label, (1,))
        advacc = acc[0].item()
        advloss = loss_robust.item()

        acc = torch_accuracy(pred2, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss_natural.item()

        param = next(net.parameters())
        grad_mean = torch.mean(param.grad)
        optimizer.step()

        pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['cleanAcc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['cleanloss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['Robloss'] = '{:.2f}'.format(advloss)
        pbar.set_postfix(pbar_dic)


def eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)

            with torch.no_grad():
                pred = net(adv_inp)
                acc = torch_accuracy(pred, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if AttackMethod is not None else 0
    return clean_accuracy.mean, adv_acc
