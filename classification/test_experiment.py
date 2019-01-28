import os
import sys
import experiment_tools
import train_classifier
from experiment_params import get_categories, ExperimentParams
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(argv):
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_dir,argv.dataset))

    # a basic experiment
    args = ExperimentParams(pattern = argv.pattern, d_out = argv.d_out,
                                seed = argv.seed, loaded_embedding = loaded_embedding,
                                dataset = argv.dataset, use_rho = False,
                                depth = argv.depth, gpu=argv.gpu,
                                batch_size=argv.batch_size,
                                base_data_dir = argv.base_dir, input_model=argv.input_model)

    if argv.visualize > 0:
        train_classifier.main_visualize(args, os.path.join(argv.base_dir,argv.dataset), argv.visualize)
    else:
        _,_,_ = train_classifier.main_test(args)

    return 0
        


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser()])
    parser.add_argument("-m", "--input_model", help="Saved model file", required=True, type=str)
    parser.add_argument("-v", "--visualize", help="Visualize (rather than test): top_k phrases to visualize", type=int, default=0)
    sys.exit(main(parser.parse_args()))
