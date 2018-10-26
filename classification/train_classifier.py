import sys
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                use_epsilon_steps=args.use_epsilon_steps
            )
            d_out = args.d_out
        else:
            assert False
        self.out = nn.Linear(d_out, nclasses)


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
            out_fwd, hidden_fwd = self.encoder(emb_fwd)
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

def log_groups(model, args, logging_file):
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
    

def init_logging(args):
    dir_path = "/home/jessedd/projects/rational-recurrences/classification/logging/"
    file_name = args.file_name() + ".txt"
    
    logging_file = open(dir_path + file_name, "w")

    logging_file.write(str(args))
    print(args)
    print("saving in {}".format(args.file_name()))
    return logging_file
    

def train_model(epoch, model, optimizer,
                train_x, train_y, valid_x, valid_y,
                test_x, test_y,
                best_valid, test_err, unchanged, scheduler, logging_file):
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

        #import pdb; pdb.set_trace()
        regularization_groups = get_regularization_groups(model, args)
        log_groups(model, args, logging_file)

        regularization_term = regularization_groups.sum()

        reg_loss = loss + args.reg_strength * regularization_term

        reg_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)


        optimizer.step()
        if args.num_epochs_debug != -1 and epoch > args.num_epochs_debug:
            import pdb; pdb.set_trace()
        logging_file.write("took {} seconds. reg_term: {}, reg_loss: {}".format(round(time.time() - iter_start_time,2),
                                                                   round(float(regularization_term),4), round(float(reg_loss),4)))
    
    valid_err = eval_model(niter, model, valid_x, valid_y)
    scheduler.step(valid_err)

    epoch_string = "\n"
    epoch_string += "-" * 110 + "\n"
    epoch_string += "| Epoch={} | iter={} | lr={} | reg_strength={} | train_loss={:.6f} | valid_err={:.6f} | regularized_loss={:.6f} |\n".format(
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
        test_err = eval_model(niter, model, test_x, test_y)
    else:
        unchanged += 1
    if unchanged >= args.patience:
        stop = True

    sys.stdout.write("\n")
    sys.stdout.flush()
    return best_valid, unchanged, test_err, stop


def main(args):
    logging_file = init_logging(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = dataloader.read_SST(args.path)
    data = train_X + valid_X + test_X

    embs = dataloader.load_embedding(args.embedding)
    emb_layer = modules.EmbeddingLayer(
        data,
        fix_emb=args.fix_embedding,
        sos=SOS,
        eos=EOS,
        embs=embs
    )

    nclasses = max(train_Y) + 1
    random_perm = list(range(len(train_X)))
    np.random.shuffle(random_perm)
    valid_x, valid_y = dataloader.create_batches(
        valid_X, valid_Y,
        args.batch_size,
        emb_layer.word2id,
        sort=True,
        gpu=args.gpu,
        sos=SOS,
        eos=EOS
    )
    test_x, test_y = dataloader.create_batches(
        test_X, test_Y,
        args.batch_size,
        emb_layer.word2id,
        sort=True,
        gpu=args.gpu,
        sos=SOS,
        eos=EOS
    )

    model = Model(args, emb_layer, nclasses)
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

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience, verbose=True)

    best_valid = 1e+8
    test_err = 1e+8
    unchanged = 0
    for epoch in range(args.max_epoch):
        np.random.shuffle(random_perm)
        train_x, train_y = dataloader.create_batches(
            train_X, train_Y,
            args.batch_size,
            emb_layer.word2id,
            perm=random_perm,
            sort=True,
            gpu=args.gpu,
            sos=SOS,
            eos=EOS
        )
        best_valid, unchanged, test_err, stop = train_model(
            epoch, model, optimizer,
            train_x, train_y,
            valid_x, valid_y,
            test_x, test_y,
            best_valid, test_err,
            unchanged, scheduler, logging_file
        )

        if stop:
            break

        if args.lr_decay > 0:
            optimizer.param_groups[0]["lr"] *= args.lr_decay


    sys.stdout.write("best_valid: {:.6f}\n".format(best_valid))
    sys.stdout.write("test_err: {:.6f}\n".format(test_err))
    sys.stdout.flush()
    logging_file.write("best_valid: {:.6f}\n".format(best_valid))
    logging_file.write("test_err: {:.6f}\n".format(test_err))
    logging_file.close()



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
