import torch
import importlib
from model import common
import matplotlib.pyplot as plt
from main import eval
import seaborn as sns
from generate_data import load_datasets
from Paras import args
from model.resNet_t import get_batch



def test(model, tasks, args):
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

        if args.model == 'resNet_t':
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

    print(str(vars(args)))
    path = args.save_path + args.model + '_' + args.mode + args.file_ext

    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    (result_t_mse, result_t_rate, result_t_ratio, result_a, spent_time, model_state_dict_res, args) \
        = torch.load(path + '.pt')

    if args.model[-4:] == 'ftrl':
        plt.plot(result_t_ratio[:, 0], '-', color='r', linewidth=0.5)
        plt.draw()
        plt.savefig('results/train_ratio.png')

    leg = ['target', 'predict']

    ratio = []
    rate = []

    case = 0
    cdf = 0
    ar = 0
    write = 1

    for i in args.train:
        i = i + 1
        sns.set_theme(context='paper', style='ticks', font='Times New Roman',
                      font_scale=1.2,
                      )
        model_state_dict = torch.load(path + '_' + str(i-1) + '_state_dict.pt')
        model.load_state_dict(model_state_dict)
        ratio_per_sample, MSE_per_sample, SUM_per_sample, LABEL_per_sample = test(model, x_te, args)

        fig, axs = plt.subplots(nrows=1, ncols=5, sharex=True,
                                sharey=True, figsize=(25, 5))

        axs[0].set_ylabel('Counts of Samples')

        name = ''
        bis = 'auto'
        ls = ['-.', ':', '--', '-', 'solid', 'dashed']
        for t in range(n_tasks):
            if case == 0:
                sns.histplot(LABEL_per_sample[t * 1000: t * 1000 + 1000], stat='count', bins=bis, ax=axs[t],
                             element='poly'
                             )
                sns.histplot(SUM_per_sample[t * 1000: t * 1000 + 1000], stat='count', bins=bis, ax=axs[t],
                             color='g', element='poly'
                             )

                axs[t].set_xlabel('Sum-rate(bps/hz) in ' + 'Scenarios ' + str(t + 1))
                name = 'rate_pdf'
            elif case == 1:
                sns.histplot(MSE_per_sample[t * 1000: t * 1000 + 1000], stat='count', bins=bis, ax=axs[t],
                             color='g', element='poly'
                             )

                axs[t].set_xlabel('MSE in ' + 'Scenarios ' + str(t + 1))
                name = 'mse_pdf'
            elif case == 2:
                sns.histplot(ratio_per_sample[t * 1000: t * 1000 + 1000], stat='count', bins=bis, ax=axs[t],
                             color='g', element='poly'
                             )

                axs[t].set_xlabel('Ratio in ' + 'Scenarios ' + str(t + 1))
                name = 'ratio_pdf'

        if args.mode != 'joint':
            axs[i - 1].set_title('Selected Model ' + str(i), color='DarkRed')
        else:
            axs[2].set_title('Joint training model ', color='DarkRed')
        plt.legend(leg, loc='upper right', prop={'size': 15})
        plt.tight_layout()
        plt.savefig('results/' + name + args.file_ext + '_model' + '_' + str(i) + '.png',
                    facecolor='w', edgecolor='w', transparent=True)  # .svg

        if cdf:
            sns.set_theme(context='paper', style='ticks', font='Times New Roman',
                          font_scale=1.2
                          )
            figc, axsc = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 6))

            kwargs = {'cumulative': True, 'linestyle': ls[5]}
            sns.distplot(ratio_per_sample[(i - 1) * 1000: (i - 1) * 1000 + 1000], bins=200,
                         hist_kws=kwargs, kde_kws=kwargs, hist=False, ax=axsc)

            plt.legend(leg, loc='upper left', prop={'size': 10})
            plt.xlim(0, 1.25)
            axsc.set_xlabel('Sum-rate approximation ratios')
            axsc.set_ylabel('Probability')
            plt.tight_layout()
            plt.savefig('results/ratio_cdf' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)

        if ar:
            result_mse, result_rate, result_ratio, tol_ratio = eval(model, x_te, args)
            sns.set_theme(context='paper', style='ticks', font='Times New Roman',
                          font_scale=1
                          )
            figa, axsa = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 5))

            axsa.xaxis.set_major_locator(plt.MultipleLocator(1))
            axsa.plot([i + 1 for i in range(5)], result_rate, marker='v')

            if args.mode != 'joint':
                axsa.set_xlabel('Selected Model ' + str(i) + " with results %.4f" % (result_rate[i-1]))
            else:
                axsa.set_xlabel('Joint Training ')
            axsa.set_ylabel('Average Sum-Rate')
            plt.ylim(0, 1.)
            plt.tight_layout()
            plt.savefig('results/Average-Sum-Rate' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)

            figb, axsb = plt.subplots(nrows=1, ncols=1, sharex=True,
                                      sharey=True, figsize=(6, 5))

            axsb.xaxis.set_major_locator(plt.MultipleLocator(1))
            axsb.plot([i + 1 for i in range(5)], result_ratio, marker='v')

            if args.mode != 'joint':
                axsb.set_xlabel('Selected Model ' + str(i)+ " with results %.4f" % (result_ratio[i-1]))
            else:
                axsb.set_xlabel('Joint Training')

            axsb.set_ylabel('Average Sum-Rate Ratio')
            plt.ylim(0, 1.)
            plt.tight_layout()
            plt.savefig('results/Average-Sum-Rate-Ratio' + args.file_ext + '_' + str(i) + '.png',
                        facecolor='w', edgecolor='w', transparent=True)

        if write:
            result_mse, result_rate, result_ratio, tol_ratio = eval(model, x_te, args)
            ratio.append(result_ratio)
            rate.append(result_rate)

    if write:
        f = open('results/records.txt', 'w+')
        f.write(str(vars(args)))
        f.write('\n')

        f.write('ratio\n')
        for i in ratio:
            f.write(str(i))
            f.write('\n')

        f.write('rate\n')
        for i in rate:
            f.write(str(i))
            f.write('\n')

        f.close()


