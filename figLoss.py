import torch
import importlib
from model import common
import matplotlib.pyplot as plt
from main import eval
import seaborn as sns
from generate_data import load_datasets
from Paras import args



if __name__ == '__main__':

    print(str(vars(args)))
    path = args.save_path + args.model + '_' + args.mode + args.file_ext

    leg = ['Ex2', 'Ex3', 'Ex4']
    ls = ['-.', '--', '-', 'solid', 'dashed']
    sns.set_theme(context='paper', style='ticks', font='Times New Roman',
                  font_scale=1
                  )


    figa, axsa = plt.subplots(nrows=1, ncols=1, sharex=True,
                              sharey=True, figsize=(8, 5.1))

    p1 = path + '_loss1.pt'
    p31 = path + '_loss31.pt'
    p41 = path + '_loss41.pt'

    d1 = torch.load(p1)
    d31 = torch.load(p31)
    d41 = torch.load(p41)

    axsa.plot(d1[0], d1[1], marker='v', markersize=7, markevery=60)
    axsa.plot(d31[0], d31[1], marker='s', markersize=5, markevery=60)
    axsa.plot(d41[0], d41[1], marker='d', markersize=5, markevery=60)


    plt.legend(leg, loc='upper right', prop={'size': 10})
    axsa.set_ylabel('函数值')
    axsa.set_xlabel('训练迭代次数/次')
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/train_loss_3.svg')
    pass





