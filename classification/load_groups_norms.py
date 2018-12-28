import sys

from experiment_params import ExperimentParams
import numpy as np
import glob

np.set_printoptions(edgeitems=3,infstr='inf',
                    linewidth=9999, nanstr='nan', precision=5,
                    suppress=True, threshold=1000, formatter=None)


def main():

    #args = ExperimentParams(pattern="4-gram", d_out="24", depth = 1,
    #                        filename_prefix = "all_cs_and_equal_rho/",
    #                        dataset = "amazon_categories/original_mix/", use_rho = False, seed=None,
    #                        dropout= 0.4703, embed_dropout= 0.0805,rnn_dropout= 0.0027,
    #                        lr= 7.285E-02, weight_decay= 7.05E-06, clip_grad= 1.52, sparsity_type = "states",
    #                        reg_strength_multiple_of_loss = 1)
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon_categories/" + "original_mix/"
    file_base += "all_cs_and_equal_rho/hparam_opt/structure_search/"
    filename_endings = ["*regstrmultofloss=0.01_*"]
    for filename_ending in filename_endings:
        filenames = glob.glob(file_base + filename_ending)
        for filename in filenames:
            from_file(filename=filename)
            #try:
            #
            #except:
            #    continue
                


def from_file(args = None, filename = None, prox = False):
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


    #if "loss=0.01_1.txt" in path:
    #    import pdb; pdb.set_trace()
        
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
            #if "Epoch=85" in line:
            #    import pdb; pdb.set_trace()

            split_line = [x for x in line.split(" ") if x != '']

            if len(split_line) != len_groups and prev_line_was_data:
                #print(line)
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
                #print(edges)
                #print(line)
                if len(edges) == len_groups:
                    prev_line_was_data = True
                    wfsas.append(edges)


        vals = vals[-1]
        vals = np.asarray(vals)


        assert vals.shape[0] == 24 # this is the number of WFSAs in the model
        assert vals.shape[1] == len_groups # this is the number of edges in each WFSA
        
            
                
    return vals, best_valid


if __name__ == "__main__":
    main()
