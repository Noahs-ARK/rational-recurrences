import numpy as np
import collections

# for entropy regularization on the rhos
def from_file(file_loc, rho_bound):
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
    #print("num_less_than_pointnine: {}".format(num_less_than_pointnine))

    return pattern, d_out, num_less_than_pointnine * 1.0 / sum(ngram_counts.values())


if __name__ == "__main__":
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/amazon/"
    file_name = "norms_adam_layers=1_lr=0.0010000_regstr=0.0100000_dout=256_dropout=0.2_pattern=4-gram_sparsity=rho_entropy.txt"
    #file_name = "norms_adam_layers=1_lr=0.0010000_regstr=0.0100000_dout=24_dropout=0.2_pattern=4-gram_sparsity=rho_entropy.txt"
    from_file(file_base + file_name)
