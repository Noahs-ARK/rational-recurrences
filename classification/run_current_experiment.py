from experiment_params import ExperimentParams, get_categories
import train_classifier
import numpy as np
import load_learned_ngrams
import time


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

def main():
    loaded_embedding = preload_embed()
    
    exp_num = 7

    
    
    # a basic experiment
    if exp_num == 0:
        args = ExperimentParams(use_rho=True, pattern="4-gram", sparsity_type = "rho_entropy", rho_sum_to_one=True,
                                reg_strength=0.01, d_out="23", lr=0.001, seed = 34159)
        train_classifier.main(args)


    # finding the largest learning rate that doesn't diverge
    elif exp_num == 1:
        lrs = np.linspace(2,0.1, 10)
        for lr in lrs:
            args = ExperimentParams(pattern="4-gram", d_out="256", trainer="sgd", max_epoch=3, lr=lr, filename_prefix="lr_tuning/")
            train_classifier.main(args)
    # for testing the effect of batch size. conclusion: larger batches require larger learning rates.
    elif exp_num == 2:
        start_time = time.time()
        batch_size = 64
        for category in ["toys_&_games/", "apparel/", "health_&_personal_care/"]:
            
            args = ExperimentParams(pattern="4-gram", d_out="24", depth = 1, filename_prefix="only_last_cs/hparam_opt/",
                                    use_last_cs=True, dataset = "amazon_categories/" + category, use_rho=False,
                                    batch_size=batch_size) # seed = None
            train_classifier.main(args)
        print("it took {} seconds with batch_size = {}".format(time.time() - start_time, batch_size))
        
    elif exp_num == 3:
        start_time = time.time()
        counter = [0]
        categories = [""]
        patterns = ["4-gram", "3-gram", "2-gram", "1-gram"]
        m = 20
        n = 5
        total_evals = len(categories) * (len(patterns) + 1) * (m+n)
        for category in categories:
            for pattern in patterns:
                train_m_then_n_models(20,5,counter, total_evals, start_time,
                                      pattern=pattern, d_out = "24", depth = 1, filename_prefix="only_last_cs/hparam_opt/",
                                      use_last_cs=True, dataset = "amazon/" + category, use_rho=False,
                                      seed=None, loaded_embedding=loaded_embedding, batch_size=32)

            train_m_then_n_models(m,n,counter, total_evals, start_time,
                                  pattern="1-gram,2-gram,3-gram,4-gram", d_out = "6,6,6,6", depth = 1,
                                  filename_prefix="only_last_cs/hparam_opt/", use_last_cs=True,
                                  dataset = "amazon/" + category, use_rho = False, seed=None,
                                  loaded_embedding = loaded_embedding, batch_size=32)
                    
    elif exp_num == 7:

        start_time = time.time()
        counter = [0]
        k = 25
        m = 20
        n = 5
        categories = get_categories() #[get_categories()[len(get_categories())-1]]
        total_evals = len(categories) * (k + m + n)
        all_reg_search_counters = []
        
        for d_out in ["24"]:#, "256"]:
            for category in categories:
                # to learn the structure
                best, reg_search_counters = train_k_models_entropy_reg(k, counter, total_evals, start_time,
                                                                       use_rho = True, pattern = "4-gram", sparsity_type = "rho_entropy",
                                                                       rho_sum_to_one=True, reg_strength = 1, d_out=d_out,
                                                                       filename_prefix="only_last_cs/hparam_opt/reg_str_search/",
                                                                       dataset = "amazon_categories/" + category, seed=None,
                                                                       loaded_embedding=loaded_embedding)
                
                all_reg_search_counters.append(reg_search_counters)


                # train and eval the learned structure
                args = train_m_then_n_models(m,n,counter, total_evals,start_time,
                                             pattern = best["learned_pattern"], d_out=best["learned_d_out"],
                                             filename_prefix="only_last_cs/hparam_opt/",
                                             dataset = "amazon_categories/" + category, use_last_cs=True, learned_structure=True,
                                             use_rho = False, seed=None, loaded_embedding=loaded_embedding)
        print("search counters:")
        for search_counter in all_reg_search_counters:
            print(search_counter)


def search_reg_str(cur_assignments, kwargs):
    starting_reg_str = kwargs["reg_strength"]
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    found_small_enough_reg_str = False
    # first search by checking that after 5 epochs, more than half aren't above .9
    kwargs["max_epoch"] = 5
    counter = 0
    while not found_small_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        print("fraction under .9: {}".format(frac_under_pointnine))
        print("")
        if frac_under_pointnine < .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] / 2.0
            if kwargs["reg_strength"] < 10**-7:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_big_lr"
        else:
            found_small_enough_reg_str = True

    found_large_enough_reg_str = False
    kwargs["max_epoch"] = 25
    while not found_large_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        print("fraction under .9: {}".format(frac_under_pointnine))
        print("")
        if frac_under_pointnine > .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] * 2.0
            if kwargs["reg_strength"] > 10**4:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_small_lr"
        else:
            found_large_enough_reg_str = True
    # to set this back to the default
    kwargs["max_epoch"] = 500
    return counter, "okay_lr"

#orders them in increasing order of lr
def get_k_sorted_hparams(k,lr_lower_bound, lr_upper_bound):
    all_assignments = []
    
    for i in range(k):
        cur = hparam_sample(lr_bounds=[lr_lower_bound,lr_upper_bound])
        all_assignments.append([cur['lr'], cur])
    all_assignments.sort()
    return [assignment[1] for assignment in all_assignments]
        

def train_k_models_entropy_reg(k,counter,total_evals,start_time,**kwargs):
    assert "reg_strength" in kwargs
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    best = {
        "assignment" : None,
        "valid_err" : 1,
        "learned_pattern" : None,
        "learned_d_out" : None,
        "frac_under_pointnine": None
        }

    reg_search_counters = []
    lr_lower_bound = 5*10**-3
    lr_upper_bound = 1.5
    all_assignments = get_k_sorted_hparams(k, lr_lower_bound, lr_upper_bound)
    for i in range(len(all_assignments)):

        valid_assignment = False
        while not valid_assignment:
            cur_assignments = all_assignments[i]
            one_search_counter, lr_judgement = search_reg_str(cur_assignments, kwargs)
            reg_search_counters.append(one_search_counter)
            if lr_judgement == "okay_lr":
                valid_assignment = True
            else:
                if lr_judgement == "too_big_lr":
                    # lower the upper bound
                    lr_upper_bound = cur_assignments['lr']
                    reverse = True
                elif lr_judgement == "too_small_lr":
                    # rase lower bound
                    lr_lower_bound = cur_assignments['lr']
                    reverse = False
                else:
                    assert False, "shouldn't be here."
                new_assignments = get_k_sorted_hparams(k-i, lr_lower_bound, lr_upper_bound)
                if reverse:
                    new_assignments.reverse()
                all_assignments[i:len(all_assignments)] = new_assignments


                

        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        
        if cur_valid_err < best["valid_err"]:
            best = {
                "assignment" : cur_assignments,
                "valid_err" : cur_valid_err,
                "learned_pattern" : learned_pattern,
                "learned_d_out" : learned_d_out,
                "frac_under_pointnine": frac_under_pointnine
            }

        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))
    return best, reg_search_counters

def train_m_then_n_models(m,n,counter, total_evals,start_time,**kwargs):
    best_assignment = None
    best_valid_err = 1
    for i in range(m):
        cur_assignments = hparam_sample()
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
        
if __name__ == "__main__":
    main()
