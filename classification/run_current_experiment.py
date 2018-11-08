from experiment_params import ExperimentParams
import train_classifier
import numpy as np



def uniform_hparam_search():
    k = 500

    params = {}
    params["lr"] = [0.5, 0.0001]
    params["reg_strength"] = [0.5, 0.00001]



    for i in range(k):
        cur_params = {}
        for param in params:
            lower_bound = params[param][0]
            upper_bound = params[param][1]
            log_lower = np.log(lower_bound)
            log_upper = np.log(upper_bound)
            point = float(np.exp(np.random.uniform(log_lower, log_upper)))
            cur_params[param] = point

        #import pdb; pdb.set_trace()        
        args = ExperimentParams(**cur_params, max_epoch=15)
        train_classifier.main(args)
    


if __name__ == "__main__":

    #uniform_hparam_search()

    exp_num = 3

    # a basic experiment
    if exp_num == 0:
        args = ExperimentParams()
        train_classifier.main(args)


    # finding the largest learning rate that doesn't diverge
    elif exp_num == 1:
        lrs = np.linspace(2,0.1, 10)
        for lr in lrs:
            args = ExperimentParams(pattern="4-gram", d_out="256", trainer="sgd", max_epoch=3, lr=lr, filename_prefix="lr_tuning/")
            train_classifier.main(args)

            
    # evaluating 1,2,3,4-gram models with equal numbers of each n-gram.
    elif exp_num == 2:
        for i in ["64,64,64,64", "6,6,6,6"]:
            for j in [1,2]:
                args = ExperimentParams(pattern="1-gram,2-gram,3-gram,4-gram", d_out=i, depth=j)
                train_classifier.main(args)

    # training with sgd, decreasing the learning rate potentially every round, starting with very high learning rate
    elif exp_num == 3:
        args = ExperimentParams(pattern="4-gram", d_out="24", trainer="sgd", lr=2, patience=100, lr_patience=0, lr_schedule_decay=0.85, filename_prefix="lr_tuning/")
        train_classifier.main(args)

    
