
import importlib
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
from generate_data import load_datasets
from model.common import SumRateLoss
from Paras import args
import matplotlib.pyplot as plt
from ftrl import ftrl

from resNet_t import get_batch

class DataProducer:
    def __init__(self, data, args):
        """
        returns: (v_x, t, v_y) 每个episode t选取batch_size大小的训练数据集，不足batch_size大小的选取所有剩下的t数据，下一轮更新t=t+1
        current: 当前已训练的数据
        permutation: 训练数据索引 (ti * n_epochs * data_size, index)
        """
        self.data = data
        self.batch_size = args.batch_size
        n_tasks = len(data)

        self.permutation = []
        for t in range(n_tasks):
            N = data[t][1].size(0)
            for _ in range(args.n_epochs):
                task_p = [[t, i] for i in range(N)]
                random.shuffle(task_p)
                self.permutation += task_p
            print("Task", t, "Samples are", N)

        self.length = len(self.permutation)
        self.current = 0
        print("total length", self.length)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]  # ti is episodes order
            j = []  # selected samples index
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]


def eval(model, tasks, args):
    """
    evaluates the model on all tasks
    """
    model.eval()
    result_mse = []
    result_rate = []
    result_ratio = []
    MSEloss = torch.nn.MSELoss()

    total_pred = 0
    total_label = 0
    for i, task in enumerate(tasks):
        t = i
        xb = task[1]
        yb = task[2]

        if args.model == 'resNet_t':
            xb, _ = get_batch(xb, yb)
        # if args.cuda:
        #     xb = xb.cuda()
        output = model(xb, t).data.cpu()
        # output = (output > 0.5).float()

        rate_loss = -SumRateLoss(xb.cpu(), output, args.noise).item()
        rate_loss_of_wmmse = - \
            SumRateLoss(xb.cpu(), yb.cpu(), args.noise).item()
        result_rate.append(rate_loss)
        result_ratio.append(rate_loss / rate_loss_of_wmmse)
        result_mse.append(MSEloss(output, yb.cpu()).item())
        total_pred += rate_loss
        total_label += rate_loss_of_wmmse

    # print('MSE:', [i for i in result_mse])
    print('ratio:', [i for i in result_ratio])
    return result_mse, result_rate, result_ratio, total_pred/total_label


def train(model_o, dataProducer, x_te, args, joint=False):
    result_t_mse = []
    result_t_rate = []
    result_t_ratio = []
    time_all = []
    result_all = []  # avg performance on all test samples
    result_spasity = []
    current_task = 0
    time_start = time.time()
    time_spent = 0
    model = model_o

    if args.model[-4:] == 'ftrl':
        opt = ftrl.FTRL(model.parameters(),
                        alpha=args.ftrl_alpha,
                        beta=args.ftrl_beta,
                        l1=args.ftrl_l1,
                        l2=args.ftrl_l2)

    for (i, (v_x, t, v_y)) in enumerate(dataProducer):
        if joint:  # joint dataset train
            if i == 0:
                v_x_acc = v_x
                v_y_acc = v_y
            else:
                v_x_acc = torch.cat((v_x_acc, v_x), 0)
                v_y_acc = torch.cat((v_y_acc, v_y), 0)

            perm_index = torch.randperm(v_x_acc.size(0))
            v_x_acc = v_x_acc[perm_index]
            v_y_acc = v_y_acc[perm_index]

            perm_index = torch.randperm(int(args.batch_size/args.step))
            v_x[perm_index] = v_x_acc[perm_index]
            v_y[perm_index] = v_y_acc[perm_index]

        if args.train is not None and t != args.train:
            continue

        if args.cuda:
            v_x = v_x.cuda()
            v_y = v_y.cuda()

        if args.train is None and current_task < t:
            print('save model ', t)
            torch.save(model.state_dict(), model.fname + '_' + str(t) + '_state_dict.pt')

        time_start = time.time()

        model.train()

        if args.model[-4:] == 'ftrl':
            model.observe(v_x, t, v_y, loss_type='MSE', x_te=x_te, x_tr=x_tr, opt=opt)
        else:
            model.observe(v_x, t, v_y, loss_type='MSE', x_te=x_te, x_tr=x_tr)

        time_end = time.time()
        time_spent = time_spent + time_end - time_start

        if(((i % args.log_every) == 0) or (t != current_task)):

            num_zeros = 0
            total_params = 0
            for m in model.modules():
                if not isinstance(m, nn.Linear):
                    continue
                num_zeros += (
                        m.weight.eq(0).sum().item() + m.bias.eq(0).sum().item()
                )
                total_params += m.weight.numel() + m.bias.numel()
            sparsity = num_zeros / total_params if total_params != 0 else 0

            result_spasity.append(sparsity)

            plt.plot(result_spasity, '-', color='b', linewidth=0.5)
            plt.draw()
            plt.savefig('results/model_spasity.png')

            res_per_t_mse, res_per_t_rate, res_per_t_ratio, res_all = eval(
                model, x_te, args)
            result_t_mse.append(res_per_t_mse)
            result_t_rate.append(res_per_t_rate)
            result_t_ratio.append(res_per_t_ratio)
            result_all.append(res_all)
            current_task = t
            time_all.append(time_spent)

    # torch.save(torch.Tensor(result_spasity), 'result_spasity.pt')

    return torch.Tensor(result_t_mse), torch.Tensor(result_t_rate), torch.Tensor(result_t_ratio), torch.Tensor(result_all), time_all


if __name__ == "__main__":

    args.cuda = True if args.cuda != 'n' else False
    if args.mini_batch_size == 0:
        args.mini_batch_size = args.batch_size  # no mini iterations

    # initialize seeds
    print("seed is", args.seed)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)  # n_tasks is a episode number

    # set up dataProducer
    dataProducer = DataProducer(x_tr, args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    model.fname = args.model + '_' + args.mode + args.file_ext
    model.fname = os.path.join(args.save_path, model.fname)

    if args.cuda:
        model.cuda()

    if args.mode == 'online':
        # run model on dataProducer
        result_t_mse, result_t_rate, result_t_ratio, result_a, spent_time = train(
            model, dataProducer, x_te, args, joint=False)
    elif args.mode == 'joint':
        # run model on entire dataset
        result_t_mse, result_t_rate, result_t_ratio, result_a, spent_time = train(
            model, dataProducer, x_te, args, joint=True)
    else:
        raise AssertionError(
            "args.mode should be one of 'online', 'joint'.")

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # print stats
    print('model name: ' + model.fname)
    print('model para: ' + str(vars(args)))
    print('spent_time: ' + str(spent_time) + 's')

    if args.train is not None and args.mode != 'joint':
        print('save model ', args.train)
        torch.save(model.state_dict(), model.fname + '_' + str(args.train) + '_state_dict.pt')
    else:
        print('save model ', len(x_tr))
        torch.save(model.state_dict(), model.fname + '_' + str(len(x_tr)) + '_state_dict.pt')
    # save all results in binary file
    torch.save((result_t_mse, result_t_rate, result_t_ratio, result_a,
                spent_time, model.state_dict(), args), model.fname + '.pt')
