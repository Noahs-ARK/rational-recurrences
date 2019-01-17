from experiment_params import ExperimentParams, get_categories
import train_classifier
import numpy as np
import time
import regularization_search_experiments


def main():
    loaded_embedding = preload_embed()
    
    exp_num = 10

    start_time = time.time()
    counter = [0]
    categories = get_categories()
    
    
    # a basic experiment
    if exp_num == 0:
        args = ExperimentParams(use_rho=True, pattern="4-gram", sparsity_type = "rho_entropy", rho_sum_to_one=True,
                                reg_strength=0.01, d_out="23", lr=0.001, seed = 34159)
        train_classifier.main(args)


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


def preload_embed():
    start = time.time()
    import dataloader
    embs =  dataloader.load_embedding("/home/jessedd/data/amazon/embedding")
    print("took {} seconds".format(time.time()-start))
    print("preloaded embeddings from amazon dataset.")
    print("")
    return embs

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
        cur_valid_err, cur_test_err = train_classifier.main(args)
        if cur_valid_err < best_valid_err:
            best_assignment = cur_assignments
            best_valid_err = cur_valid_err
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    for i in range(n):
        args = ExperimentParams(filename_suffix="_{}".format(i),**kwargs,**best_assignment)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))
    return best_assignment
        
if __name__ == "__main__":
    main()
