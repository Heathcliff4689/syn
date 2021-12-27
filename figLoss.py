import numpy
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

    leg = ['Scenario A', 'Scenario B', 'Scenario C', 'Scenario D', 'Scenario E']
    ls = ['-.', '--', '-', 'solid', 'dashed']
    sns.set_theme(context='paper', style='ticks', font='Times New Roman',
                  font_scale=1
                  )


    figa, axsa = plt.subplots(nrows=1, ncols=1, sharex=True,
                              sharey=True, figsize=(8, 5.1))

    p1 = 'results/resNet_att_ftrl_online_mimo_5.pt'
    # p31 = 'results/resNet_t_ftrl_online_mimo_5.pt'
    # p41 = 'results/resNet_att_ftrl_online_mimo_5.pt'

    d1 = torch.load(p1)
    # d31 = torch.load(p31)
    # d41 = torch.load(p41)

    x = numpy.array([range(0, 800, 2)]).reshape(400)

    axsa.plot(x, d1[2][:, 0], marker='v', markersize=7, markevery=60)
    plt.show()
    axsa.plot(x, d1[2][:, 1], marker='s', markersize=5, markevery=60)
    plt.show()
    axsa.plot(x, d1[2][:, 2], marker='d', markersize=5, markevery=60)
    plt.show()
    axsa.plot(x, d1[2][:, 3], marker='<', markersize=5, markevery=60)
    plt.show()
    axsa.plot(x, d1[2][:, 4], marker='>', markersize=5, markevery=60)
    plt.show()


    plt.legend(leg, loc='lower right', prop={'size': 10})
    axsa.set_ylabel('the Ratio of DL/WMMSE')
    axsa.set_xlabel('iterations')
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/train_ratio_5_A.svg')
    pass





