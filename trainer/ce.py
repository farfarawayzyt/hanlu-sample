# @author : ThinkPad
# @date : 2023/2/3
import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import test
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from utils import AverageMeter, save_checkpoint, create_model, accuracy, exclude_ood

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item() * 100
    else:
        return (pred == label).type(torch.FloatTensor).mean().item() * 100


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
#           model, optimizer, ema_model, scheduler):
def train(args, train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler, labeled_dataloader):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.scd:
        corrector = create_model(args).to(args.device).eval()

    if args.exclude_ood:
        ## pick pseudo-inliers
        assert args.ood_ratio > 0
        exclude_ood(args, unlabeled_dataset)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                       sampler=train_sampler(unlabeled_dataset),
                                       batch_size=args.batch_size * args.mu,
                                       num_workers=args.num_workers,
                                       drop_last=True)

    labeled_iter = iter(train_labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model

    for epoch in range(args.start_epoch, args.epochs):

        # if args.scd and epoch > args.start_fix:
        scd_acc = AverageMeter()
        model.eval()

        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x, inputs_s, _), targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_labeled_trainloader)
                (inputs_x, inputs_s, _), targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, _), label_u = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), label_u = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            # batch_size = inputs_x.shape[0]
            if args.strong_aug:
                inputs_x = inputs_s
                inputs_u_w = inputs_u_s
            if args.model == 'cr_ent':
                inputs = inputs_x
                targets = targets_x
            elif args.model == 'os_cr_ent':
                inputs = torch.cat([inputs_x, inputs_u_w])
                ood_labels = torch.masked_fill(label_u, label_u >= args.label_classes, args.label_classes)
                targets = torch.cat([targets_x, ood_labels])
            elif args.model == 'oracle_cr_ent':
                inputs = torch.cat([inputs_x, inputs_u_w])
                # ood_labels = torch.ones(inputs_u_w, dtype=torch.long) * args.label_classes
                targets = torch.cat([targets_x, label_u])
            else:
                raise AttributeError(f'Unknown CE model : {args.model}')
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            # losses_x.update(Lx.item())
            # losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            # mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.local_rank in [-1, 0]:
            out_dict = test(args, test_loader, test_model, epoch)
            for k, v in out_dict.items():
                if isinstance(v, float):
                    wandb.log({k: v, 'epoch': epoch})
            test_acc = out_dict["test_accuracy"]
            wandb.log({'train_loss': losses.avg, 'epoch': epoch})
            wandb.log({'train/2.train_loss_x': losses_x.avg, 'epoch': epoch})
            wandb.log({'train/3.train_loss_u': losses_u.avg, 'epoch': epoch})
            wandb.log({'train/4.mask': mask_probs.avg, 'epoch': epoch})
            if args.scd and epoch > args.start_fix:
                wandb.log({'scd_accuracy': scd_acc.avg, 'epoch': epoch})
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def train_linear_clf(x_train, y_train, epoch=500):
    lr_start = 0.005
    # gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.enable_grad():
        clf = nn.Linear(output_size, num_class).to(x_train.device).requires_grad_(True)
        optimizer = optim.SGD(clf.parameters(), lr=lr_start, weight_decay=5e-6)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[450], gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        batch_size = 1024
        num_batches = int(np.ceil(len(x_train) / batch_size))
        pbar = tqdm(range(epoch), desc='fitting classifier...')
        for _ in pbar:
            perm = torch.randperm(len(x_train))
            for i_batch in range(num_batches):
                idx = perm[i_batch * batch_size:(i_batch + 1) * batch_size]
                emb_batch = x_train[idx]
                label_batch = y_train[idx]
                logits_batch = clf(emb_batch)
                loss = criterion(logits_batch, label_batch)
                optimizer.zero_grad()
                loss.backward()
                losses.update(loss.item(), emb_batch.size(0))
                top1.update(count_acc(logits_batch, label_batch))
                optimizer.step()
            pbar.set_description(f'loss: {losses.avg:.4f}, acc: {top1.avg:.4f}')
            scheduler.step()
    clf.eval()
    return clf
