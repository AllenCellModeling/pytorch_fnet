"""Module for command-line 'fnet' command."""


import argparse
import os
import sys

from fnet.cli import init
from fnet.cli import predict
from fnet.cli import train_model
from fnet.utils.general_utils import init_fnet_logging


def main() -> None:
    """Main function for command-line 'fnet' command."""
    init_fnet_logging()
    parser = argparse.ArgumentParser(prog="fnet")
    subparser = parser.add_subparsers(title="command")
    parser_init = subparser.add_parser(
        "init",
        help=(
            "Initialize current directory with example fnet scripts and "
            "training options template."
        ),
    )
    parser_train = subparser.add_parser("train", help="Train a model.")
    parser_predict = subparser.add_parser("predict", help="Predict using a model.")
    init.add_parser_arguments(parser_init)
    train_model.add_parser_arguments(parser_train)
    predict.add_parser_arguments(parser_predict)

    parser_init.set_defaults(func=init.main)
    parser_train.set_defaults(func=train_model.main)
    parser_predict.set_defaults(func=predict.main)
    args = parser.parse_args()

    # Remove 'func' from args so it is not passed to target script
    func = args.func
    delattr(args, "func")
    sys.path.append(os.getcwd())
    func(args)


if __name__ == "__main__":
    main()
