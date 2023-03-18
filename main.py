import logging
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import set_model_config, \
    set_dataset, set_models, set_parser, \
    set_seed
from eval import eval_model

# import wandb

logger = logging.getLogger(__name__)
POSSIBLE_ROOT = []


def get_intern_ip_address():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_trainer(args):
    if args.model == 'open_match':
        from trainer.open_match import train
    elif args.model == 'fix_match_uniform':
        print("using: fix_match_uniform")
        from trainer.fix_match_uniform import train
    elif 'fix_match' in args.model:
        from trainer.fix_match import train
    elif 'simclr' in args.model:
        from trainer.simclr import train
    elif 'fix_contrast' in args.model:
        from trainer.fix_contrast import train
    elif 'cr_ent' in args.model:
        from trainer.ce import train
    elif 'contrast_match' in args.model:
        from trainer.contrast_match import train
    elif 'dummy_match' in args.model:
        from trainer.dummy_match import train
    elif 'meta-pl' in args.model:
        from trainer.meta_pseudo_label import train
    else:
        raise AttributeError(f'Unknown model : {args.model}')
    return train

def main():
    args = set_parser()
    global best_acc
    global best_acc_val
    args.ip_address = get_intern_ip_address()
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device
    args.root = POSSIBLE_ROOT + [args.root]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )
    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    set_model_config(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders, labeled_loader \
        = set_dataset(args)

    model, optimizer, scheduler = set_models(args)
    if args.model == 'meta-pl':
        t_model, t_optimizer, t_scheduler = set_models(args)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    args.start_epoch = 0
    # if args.resume:
    #     logger.info("==> Resuming from checkpoint..")
    #     assert os.path.isfile(
    #         args.resume), "Error: no checkpoint directory found!"
    #     args.out = os.path.dirname(args.resume)
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     args.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if args.use_ema:
    #         ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    # if args.amp:
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    model.zero_grad()
    name = f'{args.model}-{args.arch}-{args.dataset}{"-scd" if args.scd else ""}@{args.num_labeled}'
    logger.info(dict(args._get_kwargs()))
    # with wandb.init(  # Set the project where this run will be logged
    #         project="open-set-ssl-ICCV",
    #         # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #         name=name,
    #         # Track hyperparameters and run metadata
    #         config=vars(args),
    #         sync_tensorboard=True):

    if not args.eval_only:
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")
        train_method = get_trainer(args)
        if args.model == 'meta-pl':
            train_method(args, train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
                         ood_loaders, t_model, model, ema_model, t_optimizer, optimizer, t_scheduler, scheduler)
        else:
            from time import time
            before_train = time()
            train_method(args, train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
                         ood_loaders, model, optimizer, ema_model, scheduler, labeled_loader)
            after_train = time()
            print(f'trian_time: {after_train-before_train}')
    else:
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        eval_model(args, train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
                   ood_loaders, model, ema_model)


if __name__ == '__main__':
    main()
