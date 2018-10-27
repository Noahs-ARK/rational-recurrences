

class ExperimentParams:
    def __init__(self,
                 path = None,
                 embedding = None,
                 seed = 31415,
                 model = "rrnn",
                 semiring = "plus_times",
                 use_layer_norm = False,
                 use_output_gate = False,
                 use_rho = True,
                 use_epsilon_steps = True,
                 activation = "none",
                 trainer = "adam",
                 fix_embedding = True,                            
                 batch_size = 32,
                 max_epoch=100,
                 d_out=256,
                 dropout=0.2,
                 embed_dropout=0.2,
                 rnn_dropout=0.2,
                 depth=2,
                 lr=0.001,
                 lr_decay=0,
                 gpu=False,
                 eval_ite=50,
                 patience=30,
                 lr_patience=10,
                 weight_decay=1e-6,
                 clip_grad=5,
                 reg_strength=0,
                 num_epochs_debug=-1,
                 debug_run = False,
                 sparsity_type=None
    ):
        self.path = path 
        self.embedding = embedding
        self.seed = seed
        self.model = model
        self.semiring = semiring
        self.use_layer_norm = use_layer_norm
        self.use_output_gate = use_output_gate
        self.use_rho = use_rho
        self.use_epsilon_steps = use_epsilon_steps
        self.activation = activation
        self.trainer = trainer
        self.fix_embedding = fix_embedding
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.d_out = d_out
        self.dropout = dropout
        self.embed_dropout = embed_dropout
        self.rnn_dropout = rnn_dropout
        self.depth = depth
        self.lr = lr
        self.lr_decay =lr_decay
        self.gpu = gpu
        self.eval_ite = eval_ite
        self.patience = patience
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.reg_strength = reg_strength
        self.num_epochs_debug = num_epochs_debug
        self.debug_run = debug_run
        self.sparsity_type = sparsity_type

        self.current_experiment()

    # overwrites the default values with the current experiment
    def current_experiment(self):

        #self.lr = 0.4277069
        #self.trainer = "sgd"
        #self.reg_strength = 0.0005 #0.0032028
        #self.weight_decay = 0
        self.gpu = True
        self.d_out = "20,12,2,2"
        #self.depth = 1
        #self.num_epochs_debug = 10
        #self.lr=0.4856506
        #self.reg_strength=0.001
        #self.rnn_dropout = 0
        #self.embed_dropout = 0

        #self.debug_run = True
        self.pattern = "1-gram,2-gram,3-gram,4-gram"
        self.use_rho = False
        self.use_epsilon_steps = False
        self.batch_size = 16

        self.sparsity_type = "none" # possible values: edges, wfsa, none, states
        
        base_data_dir = "/home/jessedd/data/amazon"
        if self.debug_run:
            base_data_dir += "_debug"
        self.path = base_data_dir
        self.embedding = base_data_dir + "/embedding"

    def file_name(self):
        name = "norms_{}_lr={:.7f}_regstr={:.7f}_dout={}_dropout={}_pattern={}_sparsity={}".format(self.trainer,
                                                                                                   self.lr, self.reg_strength,
                                                                                                   self.d_out, self.rnn_dropout, self.pattern,
                                                                                                   self.sparsity_type)
        if not self.gpu:
            name = name + "_cpu"
        if self.debug_run:
            name = "DEBUG_" + name

        return name

    def __str__(self):
        return str(vars(self))
        
