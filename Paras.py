class Arguements:
    def __init__(self):
        # model parameters
        self.model = 'resNet_t'
        self.train = [1]

        # optimizer parameters   total_step = n_epochs * n_memories / observe_batch_size
        self.n_epochs = 2**4
        self.n_iter = 2*2
        self.n_memories = 5000  # [0, 20000]
        self.observe_batch_size = 200
        self.batch_size = self.observe_batch_size
        self.step = 5000  # others number n_memories / step
        self.lr = 1e-4  # resNet_t 1e-4 resNet_att 1e-3

        # general experiments parameters
        self.hidden_layers = '256-128-128-64-32'
        self.mode = 'online'
        self.noise = 1e-12
        self.user = 14

        # ftrl
        self.ftrl_alpha = 1.0
        self.ftrl_beta = 1.0
        self.ftrl_l1 = 1.0
        self.ftrl_l2 = 1.0

        # experiment parameters
        self.cuda = 'n'
        self.seed = 0
        self.log_every = 1 if self.model[-4:] != 'ftrl' else 1
        self.save_path = 'results/'

        # data parameters
        self.data_path = './'
        self.data_file = 'data/dataset_deepmimo_task5_21000_14_14.pt'
        self.file_ext = '_mimo_5'


args = Arguements()