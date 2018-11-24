from experiment_params import ExperimentParams, get_categories
import train_classifier
import numpy as np
import load_learned_ngrams

# amazon_categories/* have the following number of training examples:
# 1082 apparel/train
# 122 automotive/train
# 720 baby/train
# 394 beauty/train
# 20000 books/train
# 880 camera_&_photo/train
# 308 cell_phones_&_service/train
# 368 computer_&_video_games/train
# 14066 dvd/train
# 4040 electronics/train
# 168 gourmet_food/train
# 282 grocery/train
# 1208 health_&_personal_care/train
# 234 jewelry_&_watches/train
# 3298 kitchen_&_housewares/train
# 776 magazines/train
# 40 musical_instruments/train
# 11818 music/train
# 52 office_products/train
# 264 outdoor_living/train
# 734 software/train
# 858 sports_&_outdoors/train
# 12 tools_&_hardware/train
# 2056 toys_&_games/train
# 4512 video/train





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

    exp_num = 7

    # a basic experiment
    if exp_num == 0:
        args = ExperimentParams(use_rho=True, pattern="4-gram", sparsity_type = "rho_entropy", rho_sum_to_one=True, reg_strength=0.01, d_out="23", lr=0.001, seed = 34159)
        train_classifier.main(args)


    # finding the largest learning rate that doesn't diverge
    elif exp_num == 1:
        lrs = np.linspace(2,0.1, 10)
        for lr in lrs:
            args = ExperimentParams(pattern="4-gram", d_out="256", trainer="sgd", max_epoch=3, lr=lr, filename_prefix="lr_tuning/")
            train_classifier.main(args)

            
    elif exp_num == 3:
        for pattern in ["4-gram", "3-gram", "2-gram", "1-gram"]:
            for depth in [1,2]:
                for d_out in ["24", "256"]:
                    
                    args = ExperimentParams(pattern=pattern, d_out=d_out, depth=depth, filename_prefix="only_last_cs/", use_last_cs=True, lr=0.0001)
                    train_classifier.main(args)
        #exp_args_1 = {"pattern": "1-gram,2-gram,3-gram,4-gram", "d_out": "20,12,2,2", "depth":2, filename_prefix="mixed/"}
        #exp_args_2 = {"pattern": "1-gram,2-gram,3-gram,4-gram", "d_out": "10,6,2,18", "depth":1, filename_prefix="mixed/"}
        #for exp_args in [exp_args_1, exp_args_2]:
        #    args = ExperimentParams(exp_args**)
        #    train_classifier.main(args)

        
    elif exp_num == 4:
        args = ExperimentParams(pattern="4-gram", d_out = "256", depth = 1, filename_prefix="only_last_cs/", use_last_cs=True, lr=0.00025, dataset = "amazon/")
        train_classifier.main(args)


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
                    # to learn the structure
                    args = ExperimentParams(use_rho = True, pattern = "4-gram", sparsity_type = "rho_entropy",
                                            rho_sum_to_one=True, reg_strength = 0.01, d_out=d_out, lr=lr,
                                            filename_prefix="only_last_cs/", dataset = "amazon_categories/" + category)
                    #train_classifier.main(args)

                    # load learned structure from file
                    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + category
                    learned_pattern, learned_d_out, frac_under_pointnine = load_learned_ngrams.from_file(file_base + args.file_name() + ".txt")
                    print(lr, d_out, category, frac_under_pointnine)
                    
                    # train and eval the learned structure
                    #args = ExperimentParams(pattern = learned_pattern, d_out=learned_d_out, lr=lr, filename_prefix="only_last_cs/",
                    #                        dataset = "amazon_categories/" + category, use_last_cs=True, learned_structure=True,
                    #                        use_rho = False)
                    #train_classifier.main(args)



                    
            
