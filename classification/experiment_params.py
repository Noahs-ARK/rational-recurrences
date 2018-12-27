# these egories have more than 100 training instances.
def get_categories():
    #return ["apparel/", "automotive/", "baby/", "beauty/", "books/", "camera_&_photo/", "cell_phones_&_service/", "computer_&_video_games/", "dvd/", "electronics/", "gourmet_food/", "grocery/", "health_&_personal_care/", "jewelry_&_watches/", "kitchen_&_housewares/", "magazines/", "music/", "outdoor_living/", "software/", "sports_&_outdoors/", "toys_&_games/", "video/"]
    #return ["apparel/", "baby/", "beauty/", "books/", "camera_&_photo/", "cell_phones_&_service/", "computer_&_video_games/", "dvd/", "electronics/", "health_&_personal_care/", "kitchen_&_housewares/", "magazines/", "music/", "software/", "sports_&_outdoors/", "toys_&_games/", "video/"]
    #return ["camera_&_photo/","apparel/","health_&_personal_care/", "toys_&_games/", "kitchen_&_housewares/", "dvd/","books/", "original_mix/"]
    
    #return ["kitchen_&_housewares/","dvd/", "books/", "original_mix/"]
    #return ["dvd/","original_mix/"]
    #return ["kitchen_&_housewares/", "books/"]
    return ["kitchen_&_housewares/"]
    #return ["books/"]

        


class ExperimentParams:
    def __init__(self,
                 path = None,
                 embedding = None,
                 loaded_embedding = None,
                 seed = 314159,
                 model = "rrnn",
                 semiring = "plus_times",
                 use_layer_norm = False,
                 use_output_gate = False,
                 use_rho = True,
                 rho_sum_to_one = False,
                 use_last_cs = False,
                 use_epsilon_steps = False,
                 pattern = "2-gram",
                 activation = "none",
                 trainer = "adam",
                 fix_embedding = True,                            
                 batch_size = 64,
                 max_epoch=500,
                 d_out="256",
                 dropout=0.2,
                 embed_dropout=0.2,
                 rnn_dropout=0.2,
                 depth=1,
                 lr=0.001,
                 lr_decay=0,
                 lr_schedule_decay=0.5,
                 gpu=True,
                 eval_ite=50,
                 patience=30,
                 lr_patience=10,
                 weight_decay=1e-6,
                 clip_grad=5,
                 reg_strength=0,
                 reg_strength_multiple_of_loss=0,
                 reg_goal_params=False,
                 prox_step=False,
                 num_epochs_debug=-1,
                 debug_run = False,
                 sparsity_type="none",
                 filename_prefix="",
                 filename_suffix="",
                 dataset="amazon/",
                 learned_structure=False
    ):
        self.path = path 
        self.embedding = embedding
        self.loaded_embedding = loaded_embedding
        self.seed = seed
        self.model = model
        self.semiring = semiring
        self.use_layer_norm = use_layer_norm
        self.use_output_gate = use_output_gate
        self.use_rho = use_rho
        self.rho_sum_to_one = rho_sum_to_one
        self.use_last_cs = use_last_cs
        self.use_epsilon_steps = use_epsilon_steps
        self.pattern = pattern
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
        self.lr_decay = lr_decay
        self.lr_schedule_decay = lr_schedule_decay
        self.gpu = gpu
        self.eval_ite = eval_ite
        self.patience = patience
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.reg_strength = reg_strength
        self.reg_strength_multiple_of_loss = reg_strength_multiple_of_loss
        self.reg_goal_params = reg_goal_params
        self.prox_step = prox_step
        self.num_epochs_debug = num_epochs_debug
        self.debug_run = debug_run
        self.sparsity_type = sparsity_type
        self.filename_prefix = filename_prefix
        self.filename_suffix = filename_suffix
        self.dataset = dataset
        self.learned_structure = learned_structure
        
        self.current_experiment()

    # overwrites the default values with the current experiment
    def current_experiment(self):

        #self.lr = 0.4277069
        #self.trainer = "sgd"
        #self.reg_strength = 0.0005 #0.0032028
        #self.weight_decay = 0
        #self.gpu = True
        #self.d_out = "64,64,64,64" #could total 36
        #self.depth = 1
        #self.num_epochs_debug = 10
        #self.lr=0.4856506
        #self.reg_strength=0.00001
        #self.rnn_dropout = 0
        #self.embed_dropout = 0

        #self.debug_run = True
        #self.pattern = "1-gram,2-gram,3-gram,4-gram"
        #self.use_rho = False
        #self.use_last_cs = True
        #self.batch_size = 16

        #self.sparsity_type = "none" # possible values: edges, wfsa, none, states, rho_entropy
        
        base_data_dir = "/home/jessedd/data/"
        if self.debug_run:
            base_data_dir += "amazon_debug/"
        else:
            base_data_dir += self.dataset
        self.path = base_data_dir
        self.embedding = base_data_dir + "embedding"

    def filename(self):

        if self.sparsity_type == "none" and self.learned_structure:
            sparsity_name = self.learned_structure
        else:
            sparsity_name = self.sparsity_type
        if self.debug_run:
            self.filename_prefix += "DEBUG_"
        name = "{}{}_layers={}_lr={:.3E}_dout={}_drout={:.4f}_rnndout={:.4f}_embdout={:.4f}_wdecay={:.2E}_clip={:.2f}_pattern={}_sparsity={}".format(
            self.filename_prefix, self.trainer, self.depth, self.lr, self.d_out, self.dropout, self.rnn_dropout, self.embed_dropout,
            self.weight_decay, self.clip_grad, self.pattern, sparsity_name)
        if self.reg_strength > 0:
            name += "_regstr={:.3E}".format(self.reg_strength)
        if self.reg_strength_multiple_of_loss:
            name += "_regstrmultofloss={}".format(self.reg_strength_multiple_of_loss)
        if self.reg_goal_params:
            name += "_goalparams={}".format(self.reg_goal_params)
        if self.prox_step:
            name += "_prox"
        if self.filename_suffix != "":
            name += self.filename_suffix
        if not self.gpu:
            name = name + "_cpu"

        return name

    def __str__(self):
        return str(vars(self))
        
