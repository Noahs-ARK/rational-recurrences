from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
import os
from experiment_params import ExperimentParams, get_categories
import train_classifier
import numpy as np
import time
import regularization_search_experiments
import experiment_tools


def main(argv):
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_dir,argv.dataset))
    
    exp_num = 0

    start_time = time.time()
    counter = [0]
    categories = get_categories()
    
    
    # a basic experiment
    if exp_num == 0:
        args = ExperimentParams(pattern = argv.pattern, d_out = argv.d_out,
                                learned_structure = argv.learned_structure, reg_goal_params = argv.reg_goal_params,
                                filename_prefix=argv.filename_prefix,
                                seed = argv.seed, loaded_embedding = loaded_embedding,
                                dataset = argv.dataset, use_rho = False,
                                clip_grad = argv.clip, dropout = argv.dropout, rnn_dropout = argv.rnn_dropout,
                                embed_dropout = argv.embed_dropout, gpu=argv.gpu,
                                max_epoch = argv.max_epoch, patience = argv.patience,
                                batch_size = argv.batch_size, use_last_cs=argv.use_last_cs,
                                lr = argv.lr, weight_decay = argv.weight_decay, depth = argv.depth, logging_dir = argv.logging_dir,
                                base_data_dir = argv.base_dir, output_dir = argv.model_save_dir)
        cur_valid_err = train_classifier.main(args)

    # finding the largest learning rate that doesn't diverge, for evaluating the claims in this paper:
    # The Marginal Value of Adaptive Gradient Methods in Machine Learning
    # https://arxiv.org/abs/1705.08292
    # conclusion: their results don't hold for our models.
    elif exp_num == 1:
        lrs = np.linspace(2,0.1, 10)
        for lr in lrs:
            args = ExperimentParams(pattern="4-gram", d_out="256", trainer="sgd", max_epoch=3, lr=lr, filename_prefix="lr_tuning/")
            train_classifier.main(args)

    # baseline experiments for 1-gram up to 4-gram models
    elif exp_num == 3:
        patterns = ["4-gram", "3-gram", "2-gram", "1-gram"]
        m = 20
        n = 5
        total_evals = len(categories) * (len(patterns) + 1) * (m+n)

        for category in categories:
            for pattern in patterns:
                train_m_then_n_models(m,n,counter, total_evals, start_time,
                                      pattern=pattern, d_out = "24", depth = 1, filename_prefix="all_cs_and_equal_rho/hparam_opt/",
                                      dataset = "amazon_categories/" + category, use_rho=False,
                                      seed=None, loaded_embedding=loaded_embedding)

            train_m_then_n_models(m,n,counter, total_evals, start_time,
                                  pattern="1-gram,2-gram,3-gram,4-gram", d_out = "6,6,6,6", depth = 1,
                                  filename_prefix="all_cs_and_equal_rho/hparam_opt/",
                                  dataset = "amazon_categories/" + category, use_rho = False, seed=None,
                                  loaded_embedding = loaded_embedding)

    # to learn with an L_1 regularizer
    # first train with the regularizer, choose the best structure, then do hyperparameter search for that structure
    elif exp_num == 6:
        d_out = "24"
        k = 20
        l = 5
        m = 20
        n = 5
        reg_goal_params_list = [80, 60, 40, 20]
        total_evals = len(categories) * (m + n + k + l) * len(reg_goal_params_list)
        
        all_reg_search_counters = []

        for category in categories:
            for reg_goal_params in reg_goal_params_list:
                best, reg_search_counters = regularization_search_experiments.train_k_then_l_models(
                    k,l, counter, total_evals, start_time, reg_goal_params = reg_goal_params,
                    pattern = "4-gram", d_out = d_out, sparsity_type = "states",
                    use_rho = False,
                    filename_prefix="all_cs_and_equal_rho/hparam_opt/structure_search/add_reg_term_to_loss/",
                    seed=None,
                    loaded_embedding=loaded_embedding, reg_strength = 10**-6,
                    dataset = "amazon_categories/" + category)
                
                all_reg_search_counters.append(reg_search_counters)
                
                args = train_m_then_n_models(m,n,counter, total_evals, start_time,
                                             pattern = best['learned_pattern'], d_out = best["learned_d_out"],
                                             learned_structure = "l1-states-learned", reg_goal_params = reg_goal_params,
                                             filename_prefix="all_cs_and_equal_rho/hparam_opt/structure_search/add_reg_term_to_loss/",
                                             seed = None, loaded_embedding = loaded_embedding,
                                             dataset = "amazon_categories/" + category, use_rho = False)
        print("search counters:")
        for search_counter in all_reg_search_counters:
            print(search_counter)        


    # some rho_entropy experiments
    elif exp_num == 8:
        k = 20
        l = 5
        total_evals = len(categories) * (k + l)
        
        for d_out in ["24"]:#, "256"]:
            for category in categories:
                # to learn the structure, and train with the regularizer
                best, reg_search_counters = regularization_search_experiments.train_k_then_l_models(
                    k, l, counter, total_evals, start_time,
                    use_rho = True, pattern = "4-gram", sparsity_type = "rho_entropy",
                    rho_sum_to_one=True, reg_strength = 1, d_out=d_out,
                    filename_prefix="only_last_cs/hparam_opt/reg_str_search/",
                    dataset = "amazon_categories/" + category, seed=None,
                    loaded_embedding=loaded_embedding)
                
    # baseline for rho_entropy experiments
    elif exp_num == 9:
        categories = ["dvd/"]
        patterns = ["1-gram", "2-gram"] #["4-gram", "3-gram", "2-gram", "1-gram"]
        m = 20
        n = 5
        total_evals = len(categories) * (len(patterns) + 1) * (m+n)

        for category in categories:
            for pattern in patterns:
                # train and eval the learned structure
                args = train_m_then_n_models(m,n,counter, total_evals,start_time,
                                             pattern = pattern, d_out="24",
                                             filename_prefix="only_last_cs/hparam_opt/",
                                             dataset = "amazon_categories/" + category, use_last_cs=True,
                                             use_rho = False, seed=None, loaded_embedding=loaded_embedding)
                
    # baseline experiments for l1 regularization, on sst. very similar to exp_num 3
    elif exp_num == 10:
        patterns = ["4-gram", "3-gram", "2-gram", "1-gram"]
        m = 20
        n = 5
        total_evals = m * n
        for pattern in patterns:
            train_m_then_n_models(m,n,counter, total_evals, start_time,
                                  pattern=pattern, d_out = "24", depth = 1, filename_prefix="all_cs_and_equal_rho/hparam_opt/",
                                  dataset = "sst/", use_rho=False,
                                  seed=None, loaded_embedding=loaded_embedding)

        train_m_then_n_models(m,n,counter, total_evals, start_time,
                              pattern="1-gram,2-gram,3-gram,4-gram", d_out = "6,6,6,6", depth = 1,
                              filename_prefix="all_cs_and_equal_rho/hparam_opt/",
                              dataset = "sst/", use_rho = False, seed=None,
                              loaded_embedding = loaded_embedding)



# hparams to search over (from paper):
# clip_grad, dropout, learning rate, rnn_dropout, embed_dropout, l2 regularization (actually weight decay)
def hparam_sample(lr_bounds = [1.5, 10**-3]):
    assignments = {
        "clip_grad" : np.random.uniform(1.0, 5.0),
        "dropout" : np.random.uniform(0.0, 0.5),
        "rnn_dropout" : np.random.uniform(0.0, 0.5),
        "embed_dropout" : np.random.uniform(0.0, 0.5),
        "lr" : np.exp(np.random.uniform(np.log(lr_bounds[0]), np.log(lr_bounds[1]))),
        "weight_decay" : np.exp(np.random.uniform(np.log(10**-5), np.log(10**-7))),
    }

    return assignments

#orders them in increasing order of lr
def get_k_sorted_hparams(k,lr_upper_bound=1.5, lr_lower_bound=10**-3):
    all_assignments = []
    
    for i in range(k):
        cur = hparam_sample(lr_bounds=[lr_lower_bound,lr_upper_bound])
        all_assignments.append([cur['lr'], cur])
    all_assignments.sort()
    return [assignment[1] for assignment in all_assignments]

def train_m_then_n_models(m,n,counter, total_evals,start_time,**kwargs):
    best_assignment = None
    best_valid_err = 1
    all_assignments = get_k_sorted_hparams(m)
    for i in range(m):
        cur_assignments = all_assignments[i]
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err = train_classifier.main(args)
        if cur_valid_err < best_valid_err:
            best_assignment = cur_assignments
            best_valid_err = cur_valid_err
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    for i in range(n):
        args = ExperimentParams(filename_suffix="_{}".format(i),**kwargs,**best_assignment)
        cur_valid_err = train_classifier.main(args)
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))
    return best_assignment



def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--learned_structure", help="Learned structure", type=str, default="l1-states-learned")
    p.add_argument('--reg_goal_params', type=int, default = 20)
    p.add_argument('--filename_prefix', help='logging file prefix?', type=str, default="all_cs_and_equal_rho/saving_model_for_interpretability/")
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0.1943)
    p.add_argument("--rnn_dropout", help="Use RNN dropout", type=float, default=0.0805)
    p.add_argument("--embed_dropout", help="Use RNN dropout", type=float, default=0.3489)
    p.add_argument("-l", "--lr", help="Learning rate", type=float, default=2.553E-02)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=1.09)
    p.add_argument('-w', "--weight_decay", help="Weight decay", type=float, default=1.64E-06)
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model", type=str)
    p.add_argument("--logging_dir", help="Logging directory", type=str)
    p.add_argument("--max_epoch", help="Number of iterations", type=int, default=500)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    # p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    # p.add_argument("--debug", help="Debug", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))
