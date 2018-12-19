import run_current_experiment
from experiment_params import ExperimentParams, get_categories
import load_groups_norms


def run(k,m,n,counter, total_evals, start_time, **kwargs):
    best_assignment = run_current_experiment.train_m_then_n_models(m,n, counter, total_evals, start_time, **kwargs)
    args = ExperimentParams(**kwargs, **best_assignment)
    learned_d_out = load_groups_norms.from_file(args)
    kwargs["d_out"] = learned_d_out
    #kwargs["filename_prefix"] = "all_cs_and_equal_rho/hparam_opt/"
    kwargs["sparsity_type"] = "none"
    kwargs["learned_structure"] = "l1-learned"
    kwargs["pattern"] = "1-gram,2-gram,3-gram,4-gram"
    kwargs["reg_strength_multiple_of_loss"] = 0
    run_current_experiment.train_m_then_n_models(m,n,counter,total_evals,start_time, **kwargs)
    
    

