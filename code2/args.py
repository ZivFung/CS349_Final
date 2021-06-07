import argparse

def get_train_args():
    parser = argparse.ArgumentParser(f'test twitter dataset')

    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.1,
                        help='Probability of zeroing an activation in dropout layers.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                              multiple GPUs are available.')

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')


    parser.add_argument('--l2_wd',
                        type=float,
                        default=3e-7,
                        help='L2 weight decay.')

    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for initial autoencoder training. Negative means forever.')\


    parser.add_argument('--eval_steps',
                        type=int,
                        default=40000,
                        help='Number of steps between successive evaluations in ititial training.')


    parser.add_argument('--metric_name',
                        type=str,
                        default='AUC',
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=10,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='Number of workers when loading dataset.')

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        # required=False,
                        default="default",
                        help='Name to identify training or test run.')

    parser.add_argument("--train_path",
                        type=str,
                        default="../data/train_data.npz")

    parser.add_argument("--val_path",
                        type=str,
                        default="../data/val_data.npz")

    parser.add_argument("--test_path",
                        type=str,
                        default="../data/test_data.npz")

    parser.add_argument("--seed",
                        type=int,
                        default=234)

    parser.add_argument("--word_emb_file",
                        type=str,
                        default="../data/word_emb.json")

    args=parser.parse_args()

    if args.metric_name == 'Loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("Accuracy","Recall","AUC"):
        # Best checkpoint is the one that maximizes Accuracy or recall
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')
    return args