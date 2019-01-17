import sys
from experiment_params import ExperimentParams
import numpy as np
import glob

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)

def main():
    l1_or_entropy = "l1"

    if l1_or_entropy == "l1":
        l1_example()
    elif l1_or_entropy == "entropy":
        entropy_example()


# the next few functions are for l1-regularized models, below that are the entropy regularization models

def l1_example():
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + "books/"
    file_base += "all_cs_and_equal_rho/hparam_opt/structure_search/add_reg_term_to_loss/"
    filename_endings = ["*sparsity=states*goalparams=80*"]
    for filename_ending in filename_endings:
        filenames = glob.glob(file_base + filename_ending)

        for filename in filenames:
            from_file(filename=filename)

def l1_group_norms(args = None, filename = None, prox = False):
    norms, best_valid = get_norms(args, filename)

    if not prox:
        threshold = 0.1
    else:
        threshold = 0.0001
    #threshold = min(norms[-1][:,0])
    learned_ngrams = norms > threshold
    ngram_counts = [0] * (len(learned_ngrams[0]) + 1)
    weirdos = []

    for i in range(len(learned_ngrams)):
        cur_ngram = 0
        cur_weird = False
        for j in range(len(learned_ngrams[i])):
            if cur_ngram == j and learned_ngrams[i][j]:
                cur_ngram += 1
            elif cur_ngram == j and not learned_ngrams[i][j]:
                continue
            elif cur_ngram != j and learned_ngrams[i][j]:
                cur_weird = True
            elif cur_ngram != j and not learned_ngrams[i][j]:
                continue
        if cur_weird:
            weirdos.append(learned_ngrams[i])

        ngram_counts[cur_ngram] += 1
    total_params = ngram_counts[1] + 2*ngram_counts[2] + 3*ngram_counts[3] + 4*ngram_counts[4]
    print("0,1,2,3,4 grams: {}, total params: {}, num out of order: {}, {}".format(str(ngram_counts), total_params, len(weirdos), best_valid))
    
    return "{},{},{},{}".format(ngram_counts[1], ngram_counts[2], ngram_counts[3], ngram_counts[4]), total_params


def get_norms(args, filename):
    if args:
        path = "/home/jessedd/projects/rational-recurrences/classification/logging/" + args.dataset
        path += args.filename() + ".txt"
    else:
        path = filename
        
    best_valid = None
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    if "sparsity=wfsa" in path:
        vals = []
        for line in lines:
            try:
                vals.append(float(line))
            except:
                continue
    elif "sparsity=edges" in path or "sparsity=states" in path:

        if "sparsity=edges" in path:
            len_groups = 8
        else:
            len_groups = 4
        
        vals = []
        prev_line_was_data = False
        wfsas = []
        for line in lines:
            if "best_valid" in line:
                best_valid = line.strip()

            split_line = [x for x in line.split(" ") if x != '']

            if len(split_line) != len_groups and prev_line_was_data:
                prev_line_was_data = False
                vals.append(wfsas)

                wfsas = []
            else:
                edges = []
                for item in split_line:
                    try:
                        edges.append(float(item))
                    except:
                        continue
                if len(edges) == len_groups:
                    prev_line_was_data = True
                    wfsas.append(edges)

        vals = vals[-1]
        vals = np.asarray(vals)

        assert vals.shape[0] == 24 # this is the number of WFSAs in the model
        assert vals.shape[1] == len_groups # this is the number of edges in each WFSA            
                
    return vals, best_valid





# these functions are for loading the rhos from entropy regularized models


def entropy_example():
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon/"
    file_name = "norms_adam_layers=1_lr=0.0010000_regstr=0.0100000_dout=256_dropout=0.2_pattern=4-gram_sparsity=rho_entropy.txt"
    from_file(file_base + file_name)


def entropy_rhos(file_loc, rho_bound):
    backwards_lines = []
    with open(file_loc, "r") as f:
        lines = f.readlines()

        found_var = False
        for i in range(len(lines)):
            if i == 0:
                continue
            if "Variable containing:" in lines[-i]:
                break
            if found_var:
                backwards_lines.append(lines[-i].strip())
            if "[torch.cuda.FloatTensor of size" in lines[-i]:
                found_var = True

    backwards_lines = backwards_lines[:len(backwards_lines)-1]


    return extract_ngrams(backwards_lines, rho_bound)


def extract_ngrams(rhos, rho_bound):
    ngram_counts = collections.Counter()
    num_less_than_pointnine = 0
    for rho_line in rhos:
        cur_rho_line = np.fromstring(rho_line, dtype=float, sep = " ")
        if max(cur_rho_line) < rho_bound:
            num_less_than_pointnine += 1
        cur_ngram = np.argmax(cur_rho_line)
        ngram_counts[cur_ngram] = ngram_counts[cur_ngram] + 1

    pattern = ""
    d_out = ""
    for i in range(4):
        if ngram_counts[i] > 0:
            pattern = pattern + "{}-gram,".format(i+1)
            d_out = d_out + "{},".format(ngram_counts[i])
    pattern = pattern[:len(pattern)-1]
    d_out = d_out[:len(d_out)-1]

    return pattern, d_out, num_less_than_pointnine * 1.0 / sum(ngram_counts.values())


if __name__ == "__main__":
    main()
