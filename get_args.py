from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

def get_args():
    parser = ArgumentParser("HIRE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

    #training setting
    parser.add_argument("--only_test", default=False, type=bool)
    parser.add_argument("--ablation", default='all', type=str) #'all','onlyrow','onlycol','onlyfea'
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', default=200, type=int, help='seed')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Warm user ratio for training. Only used for lastfm')
    parser.add_argument('--train_states', type=str, default="warm_state", help='"warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"')
    parser.add_argument('--states', type=str, default="user_cold_state", help='"warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"')

    parser.add_argument('--num_masks', type=float, default=0.2, help='proportion of masks')

    parser.add_argument("--data_set", type=str, default="movielens") #lastfm;movielens;bookcrossing;douban
    parser.add_argument('--dataset_path', type=str, default='xxx/data') #1
    parser.add_argument('--run_dir', type=str, default='xxx/wandb')
    parser.add_argument('--task_num', type=int, default=1024)
    parser.add_argument('--val_task_num', type=int, default=128)
    parser.add_argument('--test_task_num', type=int, default=128)
    parser.add_argument('--row_num', type=int, default=32) #matrix row number
    parser.add_argument('--col_num', type=int, default=32) #matrix col number
    parser.add_argument('--batch_size', type=int, default=8) #batch size

    parser.add_argument('--support_ratio', type=float, default=0.1) #matrix col number
    parser.add_argument('--diversity_ratio', type=float, default=0.2) #matrix col number

    # used for movie datasets
    parser.add_argument('--num_gender', type=int, default=2, help='User information.')#1
    parser.add_argument('--num_age', type=int, default=7, help='User information.')#1
    parser.add_argument('--num_occupation', type=int, default=21, help='User information.')#1
    parser.add_argument('--num_zipcode', type=int, default=3402, help='User information.')#1
    parser.add_argument('--num_rate', type=int, default=6, help='Item information.')#1
    parser.add_argument('--num_genre', type=int, default=25, help='Item information.')#1
    parser.add_argument('--num_director', type=int, default=2186, help='Item information.')#1
    parser.add_argument('--num_actor', type=int, default=8030, help='Item information.')#1

    # HIRE Config
    parser.add_argument('--model_stacking_depth', dest='model_stacking_depth', type=int, default=9, help=f'Number of layers to stack.') #original 8 OOM
    parser.add_argument('--model_hidden_dropout_prob', type=float, default=0.1, help='The dropout probability for all fully connected layers in the ''(in, but not out) embeddings, attention blocks.')
    parser.add_argument('--model_layer_norm_eps', default=1e-12, type=float, help='The epsilon used by layer normalization layers.')
    parser.add_argument('--exp_gradient_clipping', type=float, default=1.,help='If > 0, clip gradients.')
    parser.add_argument('--exp_optimizer', type=str, default='lookahead_lamb',help='Model optimizer.')
    parser.add_argument('--exp_weight_decay', type=float, default=0,help='Weight decay / L2 regularization penalty.')
    parser.add_argument('--exp_lookahead_update_cadence', type=int, default=6, help='The number of steps after which Lookahead will update its ''slow moving weights with a linear interpolation between the ''slow and fast moving weights.')
    parser.add_argument('--exp_scheduler', type=str, default='flat_and_anneal', help='Learning rate scheduler: see npt/optim.py for options.')
    parser.add_argument('--exp_tradeoff', type=float, default=0.5, help='Tradeoff augmentation and label losses. If there is annealing ''(see below), this value specifies the maximum weight assigned ''to augmentation (i.e. exp_tradeoff = 1 will start by completely ''prioritizing augmentation, and gradually shift focus to labels.''total_loss = tradeoff * aug_loss + (1 - tradeoff) * label_loss')
    parser.add_argument('--exp_tradeoff_annealing', type=str, default='cosine', help='Specifies a scheduler for the tradeoff between augmentation ''and label losses. See npt/optim.py.')
    parser.add_argument('--exp_tradeoff_annealing_proportion', type=float, default=1, help='The TradeoffAnnealer will take this proportion of the total ''number of steps to complete its annealing schedule. When this ''value is set to -1, we determine this proportion by the ''exp_optimizer_warmup_proportion argument. If that argument ''is not set, we default to annealing over the total ''number of steps (which can be explicitly set with value 1).')
    parser.add_argument('--exp_optimizer_warmup_proportion', type=float, default=0.7, help='The proportion of total steps over which we warmup.''If this value is set to -1, we warmup for a fixed number of ''steps. Literature such as Evolved Transformer (So et al. 2019) ''warms up for 10K fixed steps, and decays for the rest. Can ''also be used in certain situations to determine tradeoff ''annealing, see exp_tradeoff_annealing_proportion below.')
    parser.add_argument('--exp_num_total_steps', type=float, default=100e3, help='Number of total gradient descent steps. The maximum number of ''epochs is computed as necessary using this value (e.g. in ''gradient syncing across data parallel replicates in distributed ''training).')
    #HIRE encode setting
    parser.add_argument('--embed_size', type=int, default=64, help='user/item embedding size, if they have no contents themselves')
    parser.add_argument('--embed_size_tweeting', type=int, default=16, help='user embedding size for movietweetings, if they have no contents themselves')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='used in encoder and decoder.')
    parser.add_argument("--exp_lr", default=1e-3, type=float)#1e-4-->1e-3
    parser.add_argument('--pretrain_hid_dim', type=int, default=16, help='The hidden dimension of pretrain MLP decoder.')

    #MHSA Config
    parser.add_argument('--model_rff_depth',dest='model_rff_depth',type=int,default=1, help=f'Number of layers in rFF block.')
    parser.add_argument('--model_num_heads', type=int, default=8,help='Number of attention heads. Must evenly divide model_dim_hidden.') #original 8
    parser.add_argument('--model_layer_norm_eps', default=1e-12, type=float,help='The epsilon used by layer normalization layers.''Default from BERT.')
    parser.add_argument('--model_att_score_norm', default='softmax', type=str, help='Normalization to use for the attention scores. Options include' 'softmax, constant (which divides by the sqrt of # of entries).')
    parser.add_argument('--model_att_score_dropout_prob', type=float, default=0.1,help='The dropout ratio for the attention scores.')
    parser.add_argument('--first_embedding_dim', type=int, default=32, help='Embedding dimension for item and user.')#1
    parser.add_argument('--second_embedding_dim', type=int, default=64, help='Embedding dimension for item and user.')#1


    args = parser.parse_args()

    # set the hardware parameter

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args
