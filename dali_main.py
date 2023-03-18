import os
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from utils import set_parser, set_seed, set_model_config, set_models, AverageMeter
from test_aug import strong_pipeline

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def compute_loss(args, batch_size, inputs, label_u, model, targets_x):
    logits = model(inputs)
    logits = de_interleave(logits, 2 * args.mu + 1)
    logits_x = logits[:batch_size]
    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
    # del logits
    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
    pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold)
    if 'oracle_' in args.model:
        targets_u[label_u >= args.label_classes] = label_u[label_u >= args.label_classes]
        mask = mask | (label_u >= args.label_classes)
    elif 'os_' in args.model:
        targets_u[label_u >= args.label_classes] = args.label_classes
        mask = mask | (label_u >= args.label_classes)
    mask = mask.float()
    Lu = (F.cross_entropy(logits_u_s, targets_u,
                          reduction='none') * mask).mean()
    loss = Lx + args.lambda_u * Lu
    return Lu, Lx, loss, mask, targets_u


# def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
#           model, optimizer, ema_model, scheduler):
def fixmatch_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, scheduler):
    if args.amp:
        # from torch.cuda import amp
        scaler = torch.cuda.amp.GradScaler()
        from torch.cuda.amp import autocast
    end = time.time()

    # train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # unlabeled_trainloader = DataLoader(unlabeled_dataset,
    #                                    sampler=train_sampler(unlabeled_dataset),
    #                                    batch_size=args.batch_size * args.mu,
    #                                    num_workers=args.num_workers,
    #                                    drop_last=True)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model
    # run_id = str(wandb.run.id)
    # profile_dir = f'./log/{run_id}'
    artifact = None
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #         record_shapes=True,
    #         # profile_memory=True,
    #         # with_stack=True
    #         ) as prof:
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        # batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        mask_acc = AverageMeter()

        for batch_idx in range(args.eval_step):
            # (inputs_x, _), targets_x = next(labeled_iter)

            data_start = time.perf_counter()
            labeled_data = next(labeled_iter)
            inputs_x, targets_x = labeled_data[0]['weak'], labeled_data[0]['label']
            inputs_x = torch.permute(inputs_x, (0, 3, 1, 2)).float() # NHWC => NCHW
            # targets_x = torch.squeeze(targets_x)
            targets_x = torch.squeeze(targets_x).type(torch.int64)

            # (inputs_u_w, inputs_u_s), label_u = next(unlabeled_iter)
            unlabeled_data = next(unlabeled_iter)
            inputs_u_w, inputs_u_s, label_u = unlabeled_data[0]['weak'], unlabeled_data[0]['strong'], unlabeled_data[0]['label']
            inputs_u_w = torch.permute(inputs_u_w, (0, 3, 1, 2)).float() # NHWC => NCHW
            inputs_u_s = torch.permute(inputs_u_s, (0, 3, 1, 2)).float() # NHWC => NCHW
            label_u = torch.squeeze(label_u).type(torch.int64)

            # print(inputs_x.shape, inputs_x.dtype)
            # print(targets_x.shape, targets_x.dtype)
            data_end = time.perf_counter()

            to_start = time.perf_counter()
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            to_end = time.perf_counter()
            forward_start = time.perf_counter()
            if args.amp:
                with autocast():
                    Lu, Lx, loss, mask, targets_u = compute_loss(args, batch_size, inputs, label_u, model,
                                                                targets_x)
            else:
                Lu, Lx, loss, mask, targets_u = compute_loss(args, batch_size, inputs, label_u, model, targets_x)
            forward_end = time.perf_counter()
            backward_start = time.perf_counter()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            

            scheduler.step()
            losses.update(loss.detach())
            losses_x.update(Lx.detach())
            losses_u.update(Lu.detach())
            backward_end = time.perf_counter()

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad(set_to_none=True)
            # batch_end = time.time() - end
            # batch_time.update(batch_time_)
            # mask_probs.update(mask.mean())
            # mask_acc.update((((targets_u == label_u.to(args.device)).float() * mask).sum() / (mask.sum() + 1e-6)))
            print(f"Time cost: Data: {data_end-data_start:.4f}, To: {to_end-to_start:.4f}, Forward: {forward_end-forward_start:.4f}, Backward: {backward_end-backward_start:.4f}, Total: {backward_end-data_start:.4f}")

                # prof.step()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = set_parser()

    # manually setup for one gpu
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device=torch.device('cuda', 0)

    if args.seed is not None:
        set_seed(args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    set_model_config(args)

    model, optimizer, scheduler = set_models(args)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    model.zero_grad(set_to_none=True)

    labeled_trainloader = DALIGenericIterator(
        [strong_pipeline(
            batch_size=args.batch_size,
            num_threads=4,
            device_id=0,
            file_list='filelist/imagenet_train_labeled.txt',
            file_root='data',
            initial_fill=4 * args.batch_size
        )],
        ['weak', 'strong', 'label'],
        last_batch_policy=LastBatchPolicy.DROP
    )

    unlabeled_trainloader = DALIGenericIterator(
        [strong_pipeline(
            batch_size=args.batch_size * args.mu,
            num_threads=4,
            device_id=0,
            file_list='filelist/imagenet_train_unlabeled.txt',
            file_root='data',
            initial_fill=4 * args.batch_size * args.mu
        )],
        ['weak', 'strong', 'label'],
        last_batch_policy=LastBatchPolicy.DROP
    )

    fixmatch_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, scheduler)