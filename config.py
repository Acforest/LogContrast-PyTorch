import torch
import argparse


def load_configs():
    parser = argparse.ArgumentParser(description='Command line interface for LogContrast')

    ''' Base '''
    parser.add_argument('--log_type', type=str, default='HDFS', choices=['HDFS', 'BGL', 'Thunderbird'],
                        help='The type of log dataset ["HDFS", "BGL", "Thunderbird"] (default: HDFS)')
    parser.add_argument('--model_dir', type=str, default='./models/',
                        help='The directory where LogContrast model checkpoints will be loaded or saved (default: "./models/")')
    parser.add_argument('--semantic_model_name', type=str, default='bert', choices=['bert', 'roberta', 'albert'],
                        help='The name of LogContrast semantic model ["bert", "roberta", "albert"] (default: "albert")')
    parser.add_argument('--feat_type', type=str, default='both', choices=['semantics', 'logkey', 'both'],
                        help='Feature type ["semantics", "logkey", "both"] (default: "both")')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimensions of log semantics and logkey (default: 512)')
    parser.add_argument('--vocab_size', type=int, default=120,
                        help='Vocaburary size of different kind of logkeys, recommend 120 for HDFS and 2000 for BGL (default: 2000)')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='The maximum of log key sequence length (default: 128)')

    ''' Training '''
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to perform training')
    parser.add_argument('--load_model', action='store_true',
                        help='Whether to load model for training')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--train_data_dir', type=str, default='./datasets/HDFS/HDFS_train_10000.csv',
                        help='The directory of training data (default: "./datasets/HDFS/HDFS_train_10000.csv")')
    parser.add_argument('--loss_fct', type=str, default='cl', choices=['ce', 'cl'],
                        help='Loss function used in training stage ["cl", "ce"] (default: "cl")')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs in training stage (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate for optimizer in training stage (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer in training stage (default: 0.01)')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help='Dropout rate for feature augmentation in training stage (default: 0.1)')
    parser.add_argument('--best_metric_to_save', type=str, default='f1', choices=['precision', 'recall', 'f1', 'accuracy'],
                        help='Save model based on the metric (default: "f1")')
    parser.add_argument('--lambda_cl', type=float, default=0.1,
                        help='Weight hyperparameter of contrastive loss (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature hyperparameter of contrastive loss (default: 0.5)')
    parser.add_argument('--sup_ratio', type=float, default=0.2,
                        help='The ratio of supervised learning for training (default: 0.2)')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='The ratio of noisy label for training (default: 0.0)')

    ''' Testing '''
    parser.add_argument('--do_test', action='store_true',
                        help='Whether to perform testing')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
    parser.add_argument('--test_data_dir', type=str, default='./datasets/HDFS/HDFS_test_575061.csv',
                        help='The directory of testing data (default: "./datasets/HDFS/HDFS_test_575061.csv")')
    parser.add_argument('--evo_ratio', type=float, default=0.0,
                        help='The ratio of evolution for testing logs (default: 0.0)')

    ''' Environment '''
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducibility (default: 1234)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='The device for training or testing (default: "cuda" if available else "cpu")')

    args = parser.parse_args()

    return args
