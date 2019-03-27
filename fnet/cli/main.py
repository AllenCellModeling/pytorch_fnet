from fnet.cli import train_model
from fnet.cli import init
import argparse


def main():
    """Main function for command-line 'fnet' command."""
    parser = argparse.ArgumentParser(prog='fnet')
    subparser = parser.add_subparsers(title='command')
    parser_init = subparser.add_parser(
        'init',
        help=(
            'Initialize current directory with example fnet scripts and '
            'training options template.'
        )
    )
    parser_train = subparser.add_parser('train', help='Train a model.')
    parser_predict = subparser.add_parser(
        'predict', help='Predict using a model.'
    )
    init.add_parser_arguments(parser_init)
    train_model.add_parser_arguments(parser_train)
    parser_init.set_defaults(func=init.main)
    parser_train.set_defaults(func=train_model.main)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
