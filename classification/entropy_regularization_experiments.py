def search_reg_str(cur_assignments, kwargs):
    starting_reg_str = kwargs["reg_strength"]
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    found_small_enough_reg_str = False
    # first search by checking that after 5 epochs, more than half aren't above .9
    kwargs["max_epoch"] = 1
    counter = 0
    rho_bound = .99
    while not found_small_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
        print("")
        if frac_under_pointnine < .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] / 2.0
            if kwargs["reg_strength"] < 10**-7:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_big_lr"
        else:
            found_small_enough_reg_str = True

    found_large_enough_reg_str = False
    kwargs["max_epoch"] = 5
    rho_bound = .9
    while not found_large_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
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
        
def train_k_then_l_models_entropy_reg(k,l,counter,total_evals,start_time,**kwargs):
    assert "reg_strength" in kwargs
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    best = {
        "assignment" : None,
        "valid_err" : 1,
        "learned_pattern" : None,
        "learned_d_out" : None,
        "frac_under_pointnine": None,
        "reg_strength": None
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
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt", .9)
        
        if cur_valid_err < best["valid_err"]:
            best = {
                "assignment" : cur_assignments,
                "valid_err" : cur_valid_err,
                "learned_pattern" : learned_pattern,
                "learned_d_out" : learned_d_out,
                "frac_under_pointnine": frac_under_pointnine,
                "reg_strength": kwargs["reg_strength"]
            }

        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    kwargs["filename_prefix"] = "only_last_cs/hparam_opt/"
    for i in range(l):
        kwargs["reg_strength"] = best["reg_strength"]
        args = ExperimentParams(filename_suffix="_{}".format(i),**kwargs, **best["assignment"])
        cur_valid_err, cur_test_err = train_classifier.main(args)
        counter[0] = counter[0] + 1
        
    
    return best, reg_search_counters
