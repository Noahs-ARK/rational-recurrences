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

    args = ExperimentParams(max_epoch=15)
    train_classifier.main(args)
    

