import numpy
import torch
import matplotlib
from model import common
import matplotlib.pyplot as plt
from main import eval
import seaborn as sns
from generate_data import load_datasets
from Paras import args



if __name__ == '__main__':

    print(str(vars(args)))
    path = args.save_path + args.model + '_' + args.mode + args.file_ext

    leg = ['loss', 'rate']
    ls = ['-.', '--', '-', 'solid', 'dashed']

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # sns.set_theme(context='paper', style='ticks', font='Times New Roman',
    #               font_scale=1
    #               )


    figa, axsa = plt.subplots(nrows=1, ncols=1, sharex=True,
                              sharey=True, figsize=(8, 5.1))

    p1 = 'results/resNet_att_ftrl_online_mimo_5_loss.pt'
    # p31 = 'results/resNet_t_ftrl_online_mimo_5.pt'
    p41 = 'results/resNet_att_ftrl_online_mimo_5.pt'

    d1 = torch.load(p1)
    # d31 = torch.load(p31)
    d41 = torch.load(p41)

    x = numpy.array([range(0, 800, 2)]).reshape(400)


    axsa.plot(x, d41[2][:, 1], marker='s', markersize=5, label='ratio', markevery=60, color='r')
    plt.legend(loc=7, bbox_to_anchor=(1, 0.6), prop={'size': 10})
    # plt.show()

    axsb = axsa.twinx()
    axsb.plot(d1[0], d1[1], marker='d', markersize=7, label='loss ', markevery=60)
    plt.legend(loc=7, prop={'size': 10})
    # plt.show()
    # axsa.plot(x, d31[2][:, 1], marker='s', markersize=5, markevery=60)
    # plt.show()

    # plt.legend(loc='upper right', prop={'size': 10})
    axsa.set_ylabel('谱效比值（DL/WMMSE）')
    axsb.set_ylabel('损失函数值')
    axsa.set_xlabel('训练迭代次数/次')

    # plt.tight_layout()
    plt.savefig('results/train_loss&ratio.svg')
    pass





