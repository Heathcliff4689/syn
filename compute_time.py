import time
from function_wmmse_powercontrol import WMMSE_sum_rate
import torch
import importlib
import numpy as np
from generate_data import load_datasets
from Paras import args
from resNet_t import get_batch

path = args.save_path + args.model + '_' + args.mode + args.file_ext

def test(model, tasks, args):
    model.eval()
    time_spent_nn = 0
    time_spent_wmmse = 0
    K = args.user
    ll = 0
    for i, task in enumerate(tasks):
        t = i
        xb = task[1]

        if args.model == 'resNet_t':
            xb, _ = get_batch(xb, None)

        ll = len(xb)
        t_start = time.time_ns()
        output = model(xb, t).data.cpu()
        t_end = time.time_ns()
        time_spent_nn = time_spent_nn + t_end - t_start

        xb = np.array(xb)
        Pmax = 1
        Pini = np.ones(K)
        for tmp in range(ll):
            H = np.reshape(xb[tmp, :], (K, K))
            t_start = time.time_ns()
            output = WMMSE_sum_rate(Pini, H, Pmax, args.noise)
            t_end = time.time_ns()
            time_spent_wmmse = time_spent_wmmse + t_end - t_start

    return time_spent_nn / 1e6 / ll / 5, time_spent_wmmse / 1e6 / ll / 5

if __name__ == '__main__':
    # load data
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    model_state_dict = torch.load(path + '_' + '1' + '_state_dict.pt')
    model.load_state_dict(model_state_dict)
    t1, t2 = test(model, x_te, args)

    print("DNN->", t1, "WMMSE->", t2, "DNN/WMMSE ->", t1/t2)
