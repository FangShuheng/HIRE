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
    parser.add_argument('--model_hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--model_layer_norm_eps', default=1e-12, type=float)
    parser.add_argument('--exp_gradient_clipping', type=float, default=1.0)
    parser.add_argument('--exp_optimizer', type=str, default='lookahead_lamb')
    parser.add_argument('--exp_weight_decay', type=float, default=0)
    parser.add_argument('--exp_lookahead_update_cadence', type=int, default=6)
    parser.add_argument('--exp_scheduler', type=str, default='flat_and_anneal')
    parser.add_argument('--exp_tradeoff', type=float, default=0.5)
    parser.add_argument('--exp_tradeoff_annealing', type=str, default='cosine')
    parser.add_argument('--exp_tradeoff_annealing_proportion', type=float, default=1)
    parser.add_argument('--exp_optimizer_warmup_proportion', type=float, default=0.7)
    parser.add_argument('--exp_num_total_steps', type=float, default=100e3)
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
