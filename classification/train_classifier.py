import sys
import os
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

sys.path.append("..")
import classification.dataloader as dataloader
import classification.modules as modules
from semiring import *
import rrnn



SOS, EOS = "<s>", "</s>"
class Model(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.embed_dropout)
        self.emb_layer = emb_layer
        use_tanh, use_relu, use_selu = 0, 0, 0
        if args.activation == "tanh":
            use_tanh = 1
        elif args.activation == "relu":
            use_relu = 1
        elif args.activation == "selu":
            use_selu = 1
        else:
            assert args.activation == "none"

        if args.model == "lstm":
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.d_out,
                args.depth,
                dropout=args.dropout,
                bidirectional=False
            )
            d_out = args.d_out
        elif args.model == "rrnn":
            if args.semiring == "plus_times":
                self.semiring = PlusTimesSemiring
            elif args.semiring == "max_plus":
                self.semiring = MaxPlusSemiring
            elif args.semiring == "max_times":
                self.semiring = MaxTimesSemiring
            else:
                assert False, "Semiring should either be [`plus_times`, " \
                              "`max_plus`, `max_times`], not {}".format(args.semiring)
            self.encoder = rrnn.RRNN(
                self.semiring,
                emb_layer.n_d,
                args.d_out,
                args.depth,
                pattern=args.pattern,
                dropout=args.dropout,
                rnn_dropout=args.rnn_dropout,
                bidirectional=False,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                layer_norm=args.use_layer_norm,
                use_output_gate=args.use_output_gate,
                use_rho=args.use_rho,
                rho_sum_to_one=args.rho_sum_to_one,
                use_last_cs=args.use_last_cs,
                use_epsilon_steps=args.use_epsilon_steps
            )
            d_out = args.d_out
        else:
            assert False
        out_size = sum([int(one_size) for one_size in d_out.split(",")])
        self.out = nn.Linear(out_size, nclasses)


    def init_hidden(self, batch_size):
        if self.args.model == "rrnn":
            return None
        else:
            assert False


    def forward(self, input):
        if self.args.model == "rrnn":
            input_fwd = input
            emb_fwd = self.emb_layer(input_fwd)
            emb_fwd = self.drop(emb_fwd)

            out_fwd, hidden_fwd, _ = self.encoder(emb_fwd)
            batch, length = emb_fwd.size(-2), emb_fwd.size(0)
            out_fwd = out_fwd.view(length, batch, 1, -1)
            feat = out_fwd[-1,:,0,:]
        else:
            emb = self.emb_layer(input)
            emb = self.drop(emb)

            output, hidden = self.encoder(emb)
            batch, length = emb.size(-2), emb.size(0)
            output = output.view(length, batch, 1, -1)
            feat = output[-1,:,0,:]

        feat = self.drop(feat)
        return self.out(feat)

    # Assume rrnn model
    def visualize(self, input):
        assert self.args.model == "rrnn"
        input_fwd = input
        emb_fwd = self.emb_layer(input_fwd)
#        emb_fwd = self.drop(emb_fwd)
        _, _, traces = self.encoder(emb_fwd, None, True, True)

        return traces

def eval_model(niter, model, valid_x, valid_y):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0
    total_loss = 0.0
    args = model.args
    for x, y in zip(valid_x, valid_y):
        x, y = Variable(x), Variable(y)
        if args.gpu:
            x, y = x.cuda(), y.cuda()
        x = (x)
        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.data[0] * x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()
        cnt += y.numel()
    model.train()
    return 1.0 - correct / cnt


def get_states_weights(model, args):

    embed_dim = model.emb_layer.n_d
    num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
    num_wfsas = int(args.d_out)
    
    reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)
    if len(model.encoder.rnn_lst) > 1:
        reshaped_second_layer_weights = model.encoder.rnn_lst[1].cells[0].weight.view(num_wfsas, num_wfsas, num_edges_in_wfsa)
        reshaped_weights = torch.cat((reshaped_weights, reshaped_second_layer_weights), 0)
    elif len(model.encoder.rnn_lst) > 2:
        assert False, "This regularization is only implemented for 2-layer networks."
            
    # to stack the transition and self-loops, so e.g. states[...,0] contains the transition and self-loop weights

    states = torch.cat((reshaped_weights[...,0:int(num_edges_in_wfsa/2)],
                        reshaped_weights[...,int(num_edges_in_wfsa/2):num_edges_in_wfsa]),0)
    return states

# this computes the group lasso penalty term
def get_regularization_groups(model, args):
    if args.sparsity_type == "wfsa":
        embed_dim = model.emb_layer.n_d
        num_edges_in_wfsa = model.encoder.rnn_lst[0].k
        reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
        l2_norm = reshaped_weights.norm(2, dim=0).norm(2, dim=1)
        return l2_norm
    elif args.sparsity_type == 'edges':
        return model.encoder.rnn_lst[0].weight.norm(2, dim=0)
    elif args.sparsity_type == 'states':
        states = get_states_weights(model, args)
        return states.norm(2,dim=0) # a num_wfsa by n-gram matrix
    elif args.sparsity_type == "rho_entropy":
        assert args.depth == 1, "rho_entropy regularization currently implemented for single layer networks"
        bidirectional = model.encoder.rnn_lst[0].cells[0].bidirectional
        assert not bidirectional, "bidirectional not implemented"
        
        num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
        num_wfsas = int(args.d_out)
        bias_final = model.encoder.rnn_lst[0].cells[0].bias_final
        
        sm = nn.Softmax(dim=2)
        # the 1 in the line below is for non-bidirectional models, would be 2 for bidirectional
        rho = sm(bias_final.view(1, num_wfsas, int(num_edges_in_wfsa/2)))
        entropy_to_sum = rho * rho.log() * -1
        entropy = entropy_to_sum.sum(dim=2)
        return entropy
        
        
        

def log_groups(model, args, logging_file, groups=None):
    if groups is not None:

        if args.sparsity_type == "rho_entropy":
            num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
            num_wfsas = int(args.d_out)
            bias_final = model.encoder.rnn_lst[0].cells[0].bias_final
            
            sm = nn.Softmax(dim=2)
            # the 1 in the line below is for non-bidirectional models, would be 2 for bidirectional
            rho = sm(bias_final.view(1, num_wfsas, int(num_edges_in_wfsa/2)))
            logging_file.write(str(rho))

        else:
            logging_file.write(str(groups))
    else:
        if args.sparsity_type == "wfsa":
            embed_dim = model.emb_layer.n_d
            num_edges_in_wfsa = model.encoder.rnn_lst[0].k
            reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
            l2_norm = reshaped_weights.norm(2, dim=0).norm(2, dim=1)
            logging_file.write(str(l2_norm))
            
        elif args.sparsity_type == 'edges':
            embed_dim = model.emb_layer.n_d
            num_edges_in_wfsa = model.encoder.rnn_lst[0].k
            reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
            logging_file.write(str(reshaped_weights.norm(2, dim=0)))
            #model.encoder.rnn_lst[0].weight.norm(2, dim=0)
        elif args.sparsity_type == 'states':
            assert False, "can implement this based on get_regularization_groups, but that keeps changing"
            logging_file.write(str(states.norm(2,dim=0))) # a num_wfsa by n-gram matrix
    

def init_logging(args):

    dir_path = args.logging_dir + args.dataset + "/"
    filename = args.filename() + ".txt"

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if not os.path.exists(dir_path + args.filename_prefix):
        os.mkdir(dir_path + args.filename_prefix)

    torch.set_printoptions(threshold=5000)
        
    logging_file = open(dir_path + filename, "w")

    tmp = args.loaded_embedding
    args.loaded_embedding=True
    logging_file.write(str(args))
    args.loaded_embedding = tmp
    
    #print(args)
    print("saving in {}".format(args.dataset + args.filename()))
    return logging_file


def regularization_stop(args, model):
    if args.sparsity_type == "states" and args.prox_step:
        states = get_states_weights(model, args)
        if states.norm(2,dim=0).sum().data[0] == 0:
            return True
    else:
        return False

# following https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Group_lasso
# w_g - args.reg_strength * (w_g / ||w_g||_2)
def prox_step(model, args):
    if args.sparsity_type == "states":

        states = get_states_weights(model, args)
        num_states = states.shape[2]

        embed_dim = model.emb_layer.n_d
        num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
        num_wfsas = int(args.d_out)
    
        reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)
        if len(model.encoder.rnn_lst) > 1:
            assert False, "This regularization is only implemented for 2-layer networks."
            
        first_weights = reshaped_weights[...,0:int(num_edges_in_wfsa/2)]
        second_weights = reshaped_weights[...,int(num_edges_in_wfsa/2):num_edges_in_wfsa]


        for i in range(num_wfsas):
            for j in range(num_states):
                cur_group = states[:,i,j].data
                cur_first_weights = first_weights[:,i,j].data
                cur_second_weights = second_weights[:,i,j].data
                if cur_group.norm(2) < args.reg_strength:
                    #cur_group.add_(-cur_group)
                    cur_first_weights.add_(-cur_first_weights)
                    cur_second_weights.add_(-cur_second_weights)
                else:
                    #cur_group.add_(-args.reg_strength*cur_group/cur_group.norm(2))
                    cur_first_weights.add_(-args.reg_strength*cur_first_weights/cur_group.norm(2))
                    cur_second_weights.add_(-args.reg_strength*cur_second_weights/cur_group.norm(2))
    else:
        assert False, "haven't implemented anything else"
    
def train_model(epoch, model, optimizer,
                train_x, train_y, valid_x, valid_y,
                best_valid, unchanged, scheduler, logging_file):
    model.train()
    args = model.args
    N = len(train_x)
    niter = epoch * len(train_x)
    criterion = nn.CrossEntropyLoss()
    cnt = 0
    stop = False

    import time
    for x, y in zip(train_x, train_y):
        iter_start_time = time.time()
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        if args.gpu:
            x, y = x.cuda(), y.cuda()
        x = (x)

        output = model(x)
        loss = criterion(output, y)


        if args.sparsity_type == "none":
            reg_loss = loss
            regularization_term = 0
        else:
            regularization_groups = get_regularization_groups(model, args)
            
            regularization_term = regularization_groups.sum()

            if args.reg_strength_multiple_of_loss and args.reg_strength == 0:
                args.reg_strength = loss.data[0]*args.reg_strength_multiple_of_loss/regularization_term.data[0]

            if args.prox_step:
                reg_loss = loss
            else:
                reg_loss = loss + args.reg_strength * regularization_term
            
        reg_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
        optimizer.step()

        if args.prox_step:
            prox_step(model, args)
        
        if args.num_epochs_debug != -1 and epoch > args.num_epochs_debug:
            import pdb; pdb.set_trace()

        # to log every batch's loss, and how long it took
        #logging_file.write("took {} seconds. reg_term: {}, reg_loss: {}\n".format(round(time.time() - iter_start_time,2),
        #                                                           round(float(regularization_term),4), round(float(reg_loss),4)))
    regularization_groups = get_regularization_groups(model, args)
    log_groups(model, args, logging_file, regularization_groups)

    valid_err = eval_model(niter, model, valid_x, valid_y)
    scheduler.step(valid_err)

    epoch_string = "\n"
    epoch_string += "-" * 110 + "\n"
    epoch_string += "| Epoch={} | iter={} | lr={:.5f} | reg_strength={} | train_loss={:.6f} | valid_err={:.6f} | regularized_loss={:.6f} |\n".format(
        epoch, niter,
        optimizer.param_groups[0]["lr"],
        args.reg_strength,
        loss.data[0],
        valid_err,
        reg_loss.data[0]
    )
    epoch_string += "-" * 110 + "\n"

    logging_file.write(epoch_string)
    sys.stdout.write(epoch_string)
    sys.stdout.flush()
    
    if valid_err < best_valid:
        unchanged = 0
        best_valid = valid_err
    else:
        unchanged += 1
    if unchanged >= args.patience or regularization_stop(args, model):
        stop = True
        

    sys.stdout.write("\n")
    sys.stdout.flush()
    return best_valid, unchanged, stop


def main_visualize(args, dataset_file, top_k):
    model, datasets, labels, emb_layer = main_init(args)

    model.eval()

    d, l, txt_batches = dataloader.create_batches(
        datasets[1], labels[1],
        args.batch_size,
        emb_layer.word2id,
        sort=True,
        gpu=args.gpu,
        sos=SOS,
        eos=EOS,
        get_text_batches=True
    )

    if args.gpu:
        state_dict = torch.load(args.input_model)
    else:
        state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)

    if args.gpu:
        model.to_cuda(model)

    if args.gpu:
        model.cuda()

    #top_samples = torch.zeros(len(traces[0][0])), )

    n_patts = [int(one_size) for one_size in args.d_out.split(",")]

    # top_traces = [
    #                 [ [] for x in range(length) ]
    #                 for (i,length) in enumerate(n_patts)
    #                ]


    all_traces = [[[] for x in range(i)] for i in n_patts]

    for x, txt_x in zip(d, txt_batches):
        # print(len(x[0]), len(txt_x))
        assert(len(x[0]) == len(txt_x))

        # print(x[-5][-5], x[-4][-5], x[-3][-5], x[-2][-5], x[-1][-5], txt_x[-5][-5:], txt_x[-4][-5:], txt_x[-3][-5:], txt_x[-2][-5:], txt_x[-1][-5:])
        # sys.exit(1)
        x = Variable(x)
        if args.gpu:
            x = x.cuda()
        x = (x)

        # traces1: n-patterns lengths, traces2: n_instances, traces3: n_patterns per pattern length
        traces = model.visualize(x)

        # print(len(traces), len(traces[0]), len(traces[0][0]))

        for i in range(len(n_patts)):
            for j in range(n_patts[i]):
                all_traces[i][j].extend(traces[i][j])

        # print('x={} and t (n-pattern length)={}, t0 (traces per 1st patt)={}, t1 (traces per 2nd patt)={}, t2={}, t3={},t00={} t01={} t10={}, t000={}, t100={}, t0000={}, t0001={}'.format(x.size(),
        #                                                                                                 len(traces),
        #                                                                                                 len(traces[0]),
        #                                                                                                 len(traces[1]),
        #                                                                                                 len(traces[2]),
        #                                                                                                 len(traces[3])
                                                                                                        # len(traces[0][0]),
                                                                                                        # len(traces[0][1]),
                                                                                                        # len(traces[1][0]),
                                                                                                        # len(traces[0][0][0]),
                                                                                                        # len(traces[1][0][0]),
                                                                                                        # len(traces[0][0][0][0]),
                                                                                                        # len(traces[1][0][0][0])
                                                                                                        # ))

        # print('x={} and t (n-pattern length)={}, t0 (traces per 1st patt)={}, t00={}'.format(
        #     x.size(), len(traces), len(traces[0]), len(traces[0][0])))

    #     # print(x.size(), x[0].size())
    #     for (i, same_length_traces) in enumerate(traces):
    #         # print(i)
    #         for j in range(n_patts[i]):
    #             # print("\t", j)
    #             for (doc_id, doc) in enumerate(txt_x):
    #                 # print("\t\t", doc_id)
    #                 update_trace(same_length_traces, doc_id, j, doc, top_traces[i][j])
    #
    # sys.exit(-1)

    # print(len(all_traces), len(all_traces[0]), len(all_traces[0][0]))
    # print(len(all_traces), len(all_traces[1]), len(all_traces[1][0]))

    # loop one: pattern length
    for (i, same_length_traces) in enumerate(traces):
        print("\nPattern length {}".format(i))
        # Loop two: number of patterns of each length
        for j in range(n_patts[i]):
            print("\nPattern index {}\n".format(j))
            patt_traces = traces[i][j]
            f = lambda pair: pair[0].score

            # print(f(a[0]), a[0].score)
            # print(a)
            local_top_traces = sorted(zip(patt_traces, txt_x), key=f, reverse=True)[:top_k]

            # print(top_traces)

            print_top_traces(local_top_traces)


    sys.stdout.flush()


def print_top_traces(top_traces):
    for (i, pair) in enumerate(top_traces):
        pair[0].print(i+1, pair[1])

# def update_trace(patt_trace, doc_id, patt_id, doc, trace_elements):
#     d = TraceElement(doc)
#
#     start_index =  0
#
#     while doc[start_index] == '<pad>':
#         start_index += 1
#
#     local_trace = [ patt_trace[x][doc_id][patt_id] for x in range(len(patt_trace))]
#
#     print(len(local_trace), patt_trace[0][doc_id][patt_id].numpy(), len(doc), start_index)
#     d.score = torch.FloatTensor(1).random_(0, 100)
#
#     d.u_indices = torch.LongTensor(int(len(patt_trace)/2)).random_(0, len(doc)).sort()[0]
#
#     # print("Adding d={}, u={}, done".format(d.score, d.u_indices))
#     trace_elements.append(d)

def main_test(args):
    model, datasets, labels, emb_layer = main_init(args)

    batched_datasets = []
    batched_labels = []
    for [dataset, label] in zip(datasets, labels):
        d, l = dataloader.create_batches(
        dataset, label,
        args.batch_size,
        emb_layer.word2id,
        sort=True,
        gpu=args.gpu,
        sos=SOS,
        eos=EOS
    )
        batched_datasets.append(d)
        batched_labels.append(l)


    if args.gpu:
        state_dict = torch.load(args.input_model)
    else:
        state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

    model.load_state_dict(state_dict)

    if args.gpu:
        model.to_cuda(model)

    if args.gpu:
        model.cuda()


    names = ['train', 'valid', 'test']


    errs = [eval_model(0, model, d, l) for [d,l] in zip(batched_datasets, batched_labels)]

    for [name, err] in zip(names, errs):
        sys.stdout.write("{}_err: {:.6f}\n".format(name, err))

    sys.stdout.flush()
    return errs

def main_init(args):
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = dataloader.read_SST(args.path)
    data = train_X + valid_X + test_X

    if args.loaded_embedding:
        embs = args.loaded_embedding
    else:
        embs = dataloader.load_embedding(args.embedding)
    emb_layer = modules.EmbeddingLayer(
        data,
        fix_emb=args.fix_embedding,
        sos=SOS,
        eos=EOS,
        embs=embs
    )

    nclasses = max(train_Y) + 1

    model = Model(args, emb_layer, nclasses)

    return model, [train_X, valid_X, test_X], [train_Y, valid_Y, test_Y], emb_layer


def main(args):
    logging_file = init_logging(args)
    model, datasets, labels, emb_layer = main_init(args)

    random_perm = list(range(len(datasets[0])))
    np.random.shuffle(random_perm)

    valid_x, valid_y = dataloader.create_batches(
        datasets[1], labels[1],
        args.batch_size,
        emb_layer.word2id,
        sort=True,
        gpu=args.gpu,
        sos=SOS,
        eos=EOS
    )

    if args.gpu:
        model.cuda()

    need_grad = lambda x: x.requires_grad

    if args.trainer == "adam":
        optimizer = optim.Adam(
            filter(need_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.trainer == "sgd":
        optimizer = optim.SGD(
            filter(need_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_schedule_decay, patience=args.lr_patience, verbose=True)

    best_valid = 1e+8
    unchanged = 0


    for epoch in range(args.max_epoch):
        np.random.shuffle(random_perm)
        train_x, train_y = dataloader.create_batches(
            datasets[0], labels[0],
            args.batch_size,
            emb_layer.word2id,
            perm=random_perm,
            sort=True,
            gpu=args.gpu,
            sos=SOS,
            eos=EOS
        )
        best_valid, unchanged, stop = train_model(
            epoch, model, optimizer,
            train_x, train_y,
            valid_x, valid_y,
            best_valid,
            unchanged, scheduler, logging_file
        )

        if stop:
            break

        if args.lr_decay > 0:
            optimizer.param_groups[0]["lr"] *= args.lr_decay

    if args.output_dir is not None:
        of = os.path.join(args.output_dir, "best_model.pth")
        print("Writing model to", of)
        torch.save(model.state_dict(), of)


    sys.stdout.write("best_valid: {:.6f}\n".format(best_valid))
#    sys.stdout.write("test_err: {:.6f}\n".format(test_err))
    sys.stdout.flush()
    logging_file.write("best_valid: {:.6f}\n".format(best_valid))
#    logging_file.write("test_err: {:.6f}\n".format(test_err))
    logging_file.close()
    return best_valid#, test_err



def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    argparser.add_argument("--seed", type=int, default=31415)
    argparser.add_argument("--model", type=str, default="rrnn")
    argparser.add_argument("--semiring", type=str, default="plus_times")
    argparser.add_argument("--use_layer_norm", type=str2bool, default=False)
    argparser.add_argument("--use_output_gate", type=str2bool, default=False)
    argparser.add_argument("--activation", type=str, default="none")
    argparser.add_argument("--trainer", type=str, default="adam")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--fix_embedding", type=str2bool, default=True,
                           help="if using pretrained embeddings, fix them or not during training")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d_out", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.2,
                           help="dropout intra RNN layers")
    argparser.add_argument("--embed_dropout", type=float, default=0.2,
                           help="dropout of embedding layer")
    argparser.add_argument("--rnn_dropout", type=float, default=0.2,
                           help="dropout of RNN layers")
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--gpu", type=str2bool, default=False)
    argparser.add_argument("--eval_ite", type=int, default=50)
    argparser.add_argument("--patience", type=int, default=30)
    argparser.add_argument("--lr_patience", type=int, default=10)
    argparser.add_argument("--weight_decay", type=float, default=1e-6)
    argparser.add_argument("--clip_grad", type=float, default=5)

    args = argparser.parse_args()
    return args


    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    sys.stdout.flush()

    main(args)
