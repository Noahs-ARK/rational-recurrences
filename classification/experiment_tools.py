from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import os



def preload_embed(dir_location):
    start = time.time()
    import dataloader
    embs =  dataloader.load_embedding(os.path.join(dir_location,"embedding_filtered"))
    print("took {} seconds".format(time.time()-start))
    print("preloaded embeddings from amazon dataset.")
    print("")
    return embs


def general_arg_parser():
    """ CLI args related to training and testing models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-d", '--base_dir', help="Data directory", type=str, required=True)
    p.add_argument("-a", "--dataset", help="Dataset name", type=str, required=True)
    p.add_argument("-p", "--pattern", help="Pattern specification", type=str, default="1-gram,2-gram,3-gram,4-gram")
    p.add_argument("--d_out", help="Output dimension(?)", type=str, default="0,4,0,2")
    p.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    p.add_argument('--depth', help="Depth of network", type=int, default=1)
    p.add_argument("-s", "--seed", help="Random seed", type=int, default=None)
    p.add_argument("-b", "--batch_size", help="Batch size", type=int, default=64)

    # p.add_argument("--max_doc_len",
    #                help="Maximum doc length. For longer documents, spans of length max_doc_len will be randomly "
    #                     "selected each iteration (-1 means no restriction)",
    #                type=int, default=-1)
    # p.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    # p.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)

    return p

