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
def hparam_sample():
    assignments = {
        "clip_grad" : np.random.uniform(1.0, 5.0),
        "dropout" : np.random.uniform(0.0, 0.5),
        "rnn_dropout" : np.random.uniform(0.0, 0.5),
        "embed_dropout" : np.random.uniform(0.0, 0.5),
        "lr" : np.exp(np.random.uniform(np.log(1.5), np.log(10**-3))),
        "weight_decay" : np.exp(np.random.uniform(np.log(10**-5), np.log(10**-7))),
    }

    return assignments

def main():

    loaded_embedding = preload_embed()
    
    exp_num = 3

    
    
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
        categories = [""]#["books/", "dvd/"] #["kitchen_&_housewares/","camera_&_photo/"] #["apparel/", "health_&_personal_care/", "toys_&_games/"]
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

            train_m_then_n_models(20,5,counter, total_evals, start_time,
                                  pattern="1-gram,2-gram,3-gram,4-gram", d_out = "6,6,6,6", depth = 1,
                                  filename_prefix="only_last_cs/hparam_opt/", use_last_cs=True,
                                  dataset = "amazon/" + category, use_rho = False, seed=None,
                                  loaded_embedding = loaded_embedding, batch_size=32)
                    
                                    
    elif exp_num == 5:
        for lr in [0.00025, 0.001]:
            categories = get_categories()
            for category in categories:
                for d_out in ["24", "256"]:
                    for pattern in ["4-gram", "3-gram", "2-gram", "1-gram"]:

                        args = ExperimentParams(pattern=pattern, d_out = d_out, depth = 1, filename_prefix="only_last_cs/",
                                                use_last_cs=True, lr=lr, dataset = "amazon_categories/" + category, use_rho=False)
                        train_classifier.main(args)


                args = ExperimentParams(pattern="1-gram,2-gram,3-gram,4-gram", d_out = "64,64,64,64", depth = 1,
                                        filename_prefix="only_last_cs/", use_last_cs=True, lr=lr, use_rho=False,
                                        dataset = "amazon_categories/" + category)
                train_classifier.main(args)

                args = ExperimentParams(pattern="1-gram,2-gram,3-gram,4-gram", d_out = "6,6,6,6", depth = 1,
                                        filename_prefix="only_last_cs/", use_last_cs=True, lr=lr,
                                        dataset = "amazon_categories/" + category, use_rho = False)
                train_classifier.main(args)


    elif exp_num == 7:
        categories = get_categories()
        for lr in [0.001]:#, 0.00025]:
            for d_out in ["24"]:#, "256"]:
                for category in categories:
                    for rerun_num in range(4):
                        # to learn the structure
                        args = ExperimentParams(use_rho = True, pattern = "4-gram", sparsity_type = "rho_entropy",
                                                rho_sum_to_one=True, reg_strength = 0.01, d_out=d_out, lr=lr,
                                                filename_prefix="only_last_cs/", filename_suffix="_{}".format(rerun_num),
                                                dataset = "amazon_categories/" + category, seed=None,
                                                loaded_embedding=loaded_embedding)

                        train_classifier.main(args)
                        
                        # load learned structure from file
                        file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category
                        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
                        print(lr, d_out, category, frac_under_pointnine)
                        
                        # train and eval the learned structure
                        args = ExperimentParams(pattern = learned_pattern, d_out=learned_d_out, lr=lr, filename_prefix="only_last_cs/",
                                                dataset = "amazon_categories/" + category, use_last_cs=True, learned_structure=True,
                                                use_rho = False, filename_suffix="_{}".format(rerun_num), seed=None,
                                                loaded_embedding=loaded_embedding)
                        train_classifier.main(args)

def search_reg_str(cur_reg_str, cur_assignments, **kwargs):
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category
    found_small_enough_reg_str = False
    # first search by checking that after 5 epochs, more than half aren't above .9
    kwargs["reg_strength"] = cur_reg_str
    kwargs["max_epoch"] = 5
    counter = 0
    while not found_small_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        if frac_under_pointnine < .25:
            kwargs["reg_strength"] = kwargs["reg_strength"]/2.0
        else:
            found_small_enough_reg_str = True

    found_large_enough_reg_str = False
    kwargs["max_epoch"] = 50
    while not found_large_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        if frac_under_pointnine > .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] * 2.0
        else:
            found_small_enough_reg_str = True
    # to set this back to the default
    kwargs["max_epoch"] = 500
    return kwargs["reg_strength"], counter
            
def train_m_then_n_models_entropy_reg(m,n,counter,total_evals,start_time,**kwargs):
    best = {
        "assignment" = None,
        "valid_err" = 1,
        "learned_pattern" = None,
        "learned_d_out" = None
        }
    cur_reg_str = 0.1

    reg_search_counters = []
    for i in range(m):
        cur_assignments = hparam_sample()
        cur_reg_str, one_search_counter = search_reg_str(cur_reg_str, cur_assignments, **kwargs)
        reg_search_counters.append(one_search_counter)
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
        
        if cur_valid_err < best["valid_err"]:
            best = {
                "assignment" = cur_assignments,
                "valid_err" = cur_valid_err,
                "learned_pattern" = learned_pattern,
                "learned_d_out" = learned_d_out
            }

        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    kwargs["pattern"] = best["learned_pattern"]
    kwargs["d_out"] = best["learned_d_out"]
    kwargs["use_last_cs"]=True
    kwargs["learned_structure"]=True
    kwargs["use_rho"] = False
    (pattern = learned_pattern, d_out=learned_d_out, lr=lr, filename_prefix="only_last_cs/",
                                                dataset = "amazon_categories/" + category, use_last_cs=True, learned_structure=True,
                                                use_rho = False, filename_suffix="_{}".format(rerun_num), seed=None,
                                                loaded_embedding=loaded_embedding)
    for i in range(n):
        args = ExperimentParams(filename_suffix="_{}".format(i),**kwargs,**best["assignment"])
        cur_valid_err, cur_test_err = train_classifier.main(args)
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

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
