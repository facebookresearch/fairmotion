def add_preprocess_args(parser):
    parser.add_argument(
        "--output-dir", required=True, help="Where to store pickle files."
    )
    parser.add_argument(
        "--rep",
        type=str,
        default="aa",
        help="Angle representation to convert data to",
        choices=["aa", "quat", "rotmat"],
    )
    parser.add_argument(
        "--num-observed",
        type=int,
        default=5,
        help="Number of observed poses in history",
    )
    parser.add_argument(
        "--frames-between-poses",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--file-list-folder",
        type=str,
    )
    return parser


def add_train_args(parser):
    parser.add_argument(
        "--train-preprocessed-file",
        type=str,
        help="Path to pickled file with preprocessed data",
        required=True,
    )
    parser.add_argument(
        "--valid-preprocessed-file",
        type=str,
        help="Path to pickled file with preprocessed data",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=256,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=2,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=10
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use",
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "--criterion",
        type=str,
        help="Loss function to use",
        default="mse",
        choices=["mse", "ce"],
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=0.001,
    )
    return parser
