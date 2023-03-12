import argparse

__all__ = ['set_parser']


def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch ContrastMatch Training')
    ## Computational Configurations
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    # parser.add_argument('--num_classes', type=int, default=6,
    #                     help='for cifar10')
    parser.add_argument('--model', type=str, default='open_match')
    parser.add_argument('--des', type=str, default='open_match'
                        )
    parser.add_argument('--scd', action='store_true', default=False)
    parser.add_argument('--no_bias', dest='bias', action='store_false', default=True)
    parser.add_argument('--no_sg', dest='sg', action='store_false', default=True)
    parser.add_argument('--soft_mask', action='store_true', default=False)
    parser.add_argument('--contrast_neg', action='store_true', default=False)
    parser.add_argument('--contrast_self', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='dataset name')
    ## Hyper-parameters

    # parser.add_argument('--n_labels_per_cls', type=int, default=None)
    parser.add_argument('--num-labeled', type=int, default=400,
                        help='number of labeled data per class')
    parser.add_argument('--num-val', type=int, default=50,
                        help='number of validation data per each class')
    parser.add_argument('--num-super', type=int, default=10,
                        help='number of super-class known classes cifar100: 10 or 15')
    parser.add_argument('--label_classes', type=int, default=6)
    parser.add_argument('--ood_ratio', type=float, default=1.0)
    parser.add_argument('--n_unlabels', type=int, default=20000)
    parser.add_argument('--selected_class', nargs='+', type=int, default=None)
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--strong_aug', action='store_true', default=False)
    parser.add_argument('--exclude_ood', action='store_true', default=False)
    parser.add_argument('--open_split', action='store_true', default=False)
    parser.add_argument('--vary_ratio', action='store_true', default=False)
    parser.add_argument('--dummy_classes', type=int, default=1)
    parser.add_argument('--logit_norm', action='store_true', default=False)
    parser.add_argument('--cRT', action='store_true', default=False)
    parser.add_argument('--ema_strong', action='store_true', default=False)

    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--no_test_ood', dest='test_ood', action='store_false', default=True)

    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet', 'resnet18'],
                        help='dataset name')
    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                        help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                        help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                             '1.0 for CIFAR100')
    parser.add_argument('--lambda_u', default=1.0, type=float
                        )
    parser.add_argument('--lambda_h', default=0.1, type=float
                        )
    parser.add_argument('--start_fix', default=10, type=int,
                        help='epoch to start fixmatch training')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total-steps', default=2 ** 19, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,
                        help='number of epochs to run')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--proj_head', default=False, action="store_true",
                        help='whether using projection head')
    parser.add_argument('--memory_bank', default=False, action="store_true",
                        help='whether using memory bank')
    parser.add_argument('--sim_type', default="ce", type=str,
                        help='similarity type of contrastive loss')

    ##
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--contrast_T', default=0.5, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--correct_T', default=0.1, type=float,
                        help='correct temperature')
    parser.add_argument('--lambda_scd', default=1.0, type=float,
                        help='correct temperature')

    args = parser.parse_args()
    return args
