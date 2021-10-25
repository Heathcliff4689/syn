<<<<<<< HEAD

=======
>>>>>>> f52f41c5f4c600b8d1f2be0d9c41bb3dcee75ff1

import argparse
import torch
from data.channel import generate_CSI


def load_datasets(args):
    d_tr, d_te, args = torch.load(args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = d_tr[0][2].size(1)
    print(args)
    return d_tr, d_te, n_inputs, n_outputs, len(d_tr)


if __name__ == "__main__":
    distribution = "Rayleigh-Rice-Geometry10-Geometry50"
    num_train = "20000"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--o', default='data/dataset_balance.pt', help='output file')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--distribution', default=distribution, type=str)
    parser.add_argument('--noise', default=1.0, type=float)
    parser.add_argument('--num_train', default=num_train, type=str)
    parser.add_argument('--num_test', default=1000, type=int)
    parser.add_argument('--K', default=10, type=int, help='number of user')
    args = parser.parse_args()

    tasks_tr = []
    tasks_te = []
    train_size = [int(k) for k in args.num_train.split('-')]
    data_distribution = args.distribution.split('-')
    assert len(train_size) == 1 or len(train_size) == len(
        data_distribution), "len mismatch"
    for t in range(len(data_distribution)):
        dist = data_distribution[t]
        num_train = train_size[0] if len(train_size) == 1 else train_size[t]

        Xtrain, Ytrain = generate_CSI(
            args.K, num_train, args.seed, dist, args.noise)
        Xtrain = torch.from_numpy(Xtrain).float()
        Ytrain = torch.from_numpy(Ytrain).float()
        tasks_tr.append([dist, Xtrain.clone(), Ytrain.clone()])

        Xtest, Ytest = generate_CSI(
            args.K, args.num_test, args.seed+2020, dist, args.noise)
        Xtest = torch.from_numpy(Xtest).float()
        Ytest = torch.from_numpy(Ytest).float()
        tasks_te.append([dist, Xtest.clone(), Ytest.clone()])

    torch.save([tasks_tr, tasks_te, args], args.o)
