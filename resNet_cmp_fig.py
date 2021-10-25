import torch
import importlib
from model import common
import matplotlib.pyplot as plt
from main import eval
import seaborn as sns
from generate_data import load_datasets
from Paras import args
from resNet_t import get_batch

def test(model, tasks, args, flag):
    model.eval()
    MSE_per_sample = []
    SUM_per_sample = []
    ratio_per_sample = []
    for i, task in enumerate(tasks):
        t = i
        xb = task[1]
        yb = task[2]
        # if args.cuda:
        #     xb = xb.cuda()

        if flag == 'resNet_t':
            xb, _ = get_batch(xb, yb)

        output = model(xb, t).data.cpu()

        predict_sumrate = common.SumRateLoss(
            xb.cpu(), output, args.noise, persample=True)
        predict_mse = torch.sum((yb.cpu() - output) ** 2, 1)
        label_sumrate = common.SumRateLoss(
            xb.cpu(), yb.cpu(), args.noise, persample=True)
        ratio = torch.div(predict_sumrate, label_sumrate)
        if i == 0:
            ratio_per_sample = ratio
            MSE_per_sample = predict_mse
            SUM_per_sample = predict_sumrate
            LABEL_per_sample = label_sumrate
        else:
            ratio_per_sample = torch.cat((ratio_per_sample, ratio), 0)
            MSE_per_sample = torch.cat((MSE_per_sample, predict_mse), 0)
            SUM_per_sample = torch.cat((SUM_per_sample, predict_sumrate), 0)
            LABEL_per_sample = torch.cat((LABEL_per_sample, label_sumrate), 0)
    return ratio_per_sample, MSE_per_sample, -SUM_per_sample, -LABEL_per_sample

if __name__ == '__main__':
    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)

    leg = ['resNet_t', 'resNet_t']
    leg1 = ['resNet_t_2', 'resNet_t_8']

    model0 = leg[0]
    model1 = leg[1]

    path0 = args.save_path + model0 + args.mode + args.file_ext
    path1 = args.save_path + model1 + args.mode + args.file_ext

    # load model
    Model = importlib.import_module('model.' + model0)
    model0 = Model.Net(n_inputs, n_outputs, n_tasks, args)

    Model = importlib.import_module('model.' + model1)
    model1 = Model.Net(n_inputs, n_outputs, n_tasks, args)

    for i in range(1, n_tasks + 1):
        model_state_dict = torch.load(path0 + '_' + str(i) + '_state_dict.pt')
        model0.load_state_dict(model_state_dict)

        model_state_dict = torch.load(path1 + '_' + str(i) + '_state_dict.pt')
        model1.load_state_dict(model_state_dict)

        ratio_per_sample0, MSE_per_sample0, SUM_per_sample0, LABEL_per_sample0 = test(model0, x_te, args, leg[0])
        ratio_per_sample1, MSE_per_sample1, SUM_per_sample1, LABEL_per_sample1 = test(model1, x_te, args, leg[1])

        cdf = 0
        ar = 1

        ls = ['-.', ':', '--', '-', 'solid', 'dashed']

        if cdf:
            sns.set_theme(context='notebook', style='whitegrid', palette=sns.color_palette('hls', 8), font='sans-serif',
                          color_codes=True, rc=None)
            figc, axsc = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 6))

            kwargs = {'cumulative': True, 'linestyle': ls[5]}
            sns.distplot(ratio_per_sample0[(i - 1) * 1000: (i - 1) * 1000 + 1000], bins=200,
                         hist_kws=kwargs, kde_kws=kwargs, hist=False, ax=axsc, color='Deepskyblue')
            sns.distplot(ratio_per_sample1[(i - 1) * 1000: (i - 1) * 1000 + 1000], bins=200,
                         hist_kws=kwargs, kde_kws=kwargs, hist=False, ax=axsc)

            plt.legend(leg1, loc='upper left', prop={'size': 10})
            plt.xlim(0.2, 1.25)
            axsc.set_xlabel('Sum-rate approximation ratios')
            axsc.set_ylabel('Probability')
            plt.tight_layout()
            plt.savefig('results/ratio_cdf' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)

        if ar:
            result_mse0, result_rate0, result_ratio0, tol_ratio0 = eval(model0, x_te, args)
            result_mse1, result_rate1, result_ratio1, tol_ratio1 = eval(model1, x_te, args)
            sns.set_theme(context='notebook', style='whitegrid', palette=sns.color_palette('hls', 8), font='sans-serif',
                          color_codes=True, rc=None)
            figa, axsa = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 5))

            axsa.xaxis.set_major_locator(plt.MultipleLocator(1))
            axsa.plot([i + 1 for i in range(5)], result_rate0, marker='v', color='Deepskyblue', linestyle=ls[4])
            axsa.plot([i + 1 for i in range(5)], result_rate1, marker='x', color='Crimson', linestyle=ls[4])

            if args.mode != 'joint':
                axsa.set_xlabel('Model ' + str(i) + '   ' + leg[0] + ' %.4f ' % result_rate0[i - 1] + ' ' + leg[1] + " %.4f"
                                % (result_rate1[i - 1]))
            else:
                axsa.set_xlabel('Joint Training')

            plt.legend(leg1, loc='upper right', prop={'size': 10})
            axsa.set_ylabel('Average Sum-Rate')
            plt.ylim(0, 1.25)
            plt.tight_layout()
            plt.savefig('results/Average-Sum-Rate' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)


            figb, axsb = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 5))

            axsb.xaxis.set_major_locator(plt.MultipleLocator(1))
            axsb.plot([i + 1 for i in range(5)], result_ratio0, marker='v', color='Deepskyblue', linestyle=ls[4])
            axsb.plot([i + 1 for i in range(5)], result_ratio1, marker='x', color='Crimson', linestyle=ls[4])

            if args.mode != 'joint':
                axsb.set_xlabel('Model ' + str(i) + '   ' + leg[0] + ' %.4f,' % result_ratio0[i - 1] + ' ' + leg[1] + " %.4f"
                                % result_ratio1[i - 1])
            else:
                axsb.set_xlabel('Joint Training')

            plt.legend(leg1, loc='upper right', prop={'size': 10})
            axsb.set_ylabel('Average Sum-Rate Ratio')
            plt.ylim(0, 1.25)
            plt.tight_layout()
            plt.savefig('results/Average-Sum-Rate-Ratio' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)
