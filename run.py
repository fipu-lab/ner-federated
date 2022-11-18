import argparse
from train import do_train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="Model name to be used in simulations.")
    parser.add_argument("--lr",
                        required=True,
                        type=str,
                        help="Dict with client and server learning rate or a float number for identical learning rates")
    parser.add_argument("--dataset",
                        type=str,
                        help="Dataset name (conll/few_nerd)")
    parser.add_argument("--num_clients",
                        default=100,
                        type=int,
                        help="Number of clients to divide the dataset to.")
    parser.add_argument("--num_train_clients",
                        default=10,
                        type=int,
                        help="Number of clients trained in each round.")
    parser.add_argument("--pretrained",
                        action='store_true',
                        help="Set this flag for use of the pretrained model weights (default false)")
    parser.add_argument("--frozen_bert",
                        action='store_true',
                        help="Set this flag to freeze the bert weights when training (default false)")
    parser.add_argument("--seq_len",
                        default=128,
                        type=int,
                        help="Sequence length (default 128)")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size (default 32)")
    parser.add_argument("--parallel_clients",
                        default=None,
                        type=int,
                        help="Max number of clients to train in parallel. By default all clients are trained in parallel.")
    parser.add_argument("--devices",
                        default='GPUS',
                        type=str,
                        help="Device on which simulations are run. Possibilities: 'CPU', 'GPU:0', 'GPU:1', 'GPUS'.")
    parser.add_argument("--is_notebook",
                        action='store_true',
                        help="Set this flag if the code is run using a notebook (default false)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    lr = args.lr
    try:
        lrs = float(lr)
        lr = {'server': lrs, 'client': lrs}
    except ValueError:
        pass
    do_train(args.model_name,
             lr=lr,
             dataset=args.dataset,
             num_clients=args.num_clients,
             num_train_clients=args.num_train_clients,
             pretrained=args.pretrained,
             frozen_bert=args.frozen_bert,
             seq_len=args.seq_len,
             batch_size=args.batch_size,
             parallel_clients=args.parallel_clients,
             devices=args.devices,
             is_notebook=args.is_notebook)
