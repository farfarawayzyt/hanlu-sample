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

from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from utils import AverageMeter, save_checkpoint, create_model, accuracy, exclude_ood,test

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def kd_loss(outputs, teacher_outputs, T=1.0):
    KD_loss = F.kl_div(F.log_softmax(outputs, dim=1),
                       F.softmax(teacher_outputs / T, dim=1))
    return KD_loss


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

    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak

    if args.exclude_ood:
        ## pick pseudo-inliers
        assert args.ood_ratio > 0
        exclude_ood(args, unlabeled_dataset)

    labeled_dataset = copy.deepcopy(train_labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    train_labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
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
                (inputs_x, _, _), targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_labeled_trainloader)
                (inputs_x, _, _), targets_x = labeled_iter.next()
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

            label_u = label_u.to(args.device)
            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
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
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            wandb.log({'train_loss': losses.avg, 'epoch': epoch})
            wandb.log({'train/2.train_loss_x': losses_x.avg, 'epoch': epoch})
            wandb.log({'train/3.train_loss_u': losses_u.avg, 'epoch': epoch})
            wandb.log({'train/4.mask': mask_probs.avg, 'epoch': epoch})
            wandb.log({'test_accuracy': test_acc, 'epoch': epoch})
            wandb.log({'test/2.test_loss': test_loss, 'epoch': epoch})
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


# def test(args, test_loader, model, epoch):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     if not args.no_progress:
#         test_loader = tqdm(test_loader,
#                            disable=args.local_rank not in [-1, 0])
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             data_time.update(time.time() - end)
#             model.eval()
#
#             inputs = inputs.to(args.device)
#             targets = targets.to(args.device)
#             outputs = model(inputs)
#
#             known_targets = targets < int(outputs.size(1))  # [0]
#             known_pred = outputs[known_targets]
#             known_targets = targets[known_targets]
#
#             loss = F.cross_entropy(known_pred, known_targets)
#
#             prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
#             losses.update(loss.item(), known_pred.shape[0])
#             top1.update(prec1.item(), known_pred.shape[0])
#             top5.update(prec5.item(), known_pred.shape[0])
#             batch_time.update(time.time() - end)
#             end = time.time()
#             if not args.no_progress:
#                 test_loader.set_description(
#                     "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
#                         batch=batch_idx + 1,
#                         iter=len(test_loader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
#                         loss=losses.avg,
#                         top1=top1.avg,
#                         top5=top5.avg,
#                     ))
#         if not args.no_progress:
#             test_loader.close()
#
#     logger.info("top-1 acc: {:.2f}".format(top1.avg))
#     logger.info("top-5 acc: {:.2f}".format(top5.avg))
#     return losses.avg, top1.avg


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
