# @author : ThinkPad 
# @date : 2023/2/17

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
from utils import AverageMeter, save_checkpoint, create_model, accuracy, exclude_ood, test
from torch import distributed as dist

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def train(args, labeled_loader, unlabeled_dataset, test_loader, val_loader, ood_loaders,
          teacher_model, student_model, avg_student_model,
          t_optimizer, s_optimizer, t_scheduler, s_scheduler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")
    criterion = nn.CrossEntropyLoss()
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    unlabeled_loader = DataLoader(unlabeled_dataset,
                                  sampler=train_sampler(unlabeled_dataset),
                                  batch_size=args.batch_size * args.mu,
                                  num_workers=args.num_workers,
                                  drop_last=True)
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    args.best_top1 = 0
    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)
    args.total_steps = args.epochs * args.eval_step
    for step in range(args.total_steps):
        epoch = step // args.eval_step
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            (images_l, _, _), targets = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            (images_l, _, _), targets = next(labeled_iter)

        try:
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()
            (images_uw, images_us, _), _ = next(unlabeled_iter)
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()
            (images_uw, images_us, _), _ = next(unlabeled_iter)

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        targets = targets.to(args.device)

        batch_size = images_l.shape[0]
        t_images = torch.cat((images_l, images_uw, images_us))
        t_logits = teacher_model(t_images)
        t_logits_l = t_logits[:batch_size]
        t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
        del t_logits

        t_loss_l = criterion(t_logits_l, targets)

        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.T, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        t_loss_u = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
        )
        weight_u = args.lambda_u * min(1., (step + 1))
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        s_images = torch.cat((images_l, images_us))
        s_logits = student_model(s_images)
        s_logits_l = s_logits[:batch_size]
        s_logits_us = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
        s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_loss.backward()
        s_optimizer.step()
        # s_scaler.update()
        s_scheduler.step()
        if args.use_ema:
            avg_student_model.update(student_model)

        with torch.no_grad():
            s_logits_l = student_model(images_l)
        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

        # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
        # dot_product = s_loss_l_old - s_loss_l_new

        # author's code formula
        dot_product = s_loss_l_new - s_loss_l_old
        # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        # dot_product = dot_product - moving_dot_product

        _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
        # test
        # t_loss_mpl = torch.tensor(0.).to(args.device)
        t_loss = t_loss_uda + t_loss_mpl

        t_loss.backward()
        t_optimizer.step()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
            # f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        # if args.local_rank in [-1, 0]:
        #     args.writer.add_scalar("lr", get_lr(s_optimizer), step)
        #             wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                #                 wandb.log({"train/1.s_loss": s_losses.avg,
                #                            "train/2.t_loss": t_losses.avg,
                #                            "train/3.t_labeled": t_losses_l.avg,
                #                            "train/4.t_unlabeled": t_losses_u.avg,
                #                            "train/5.t_mpl": t_losses_mpl.avg,
                #                            "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss, top1 = test(args, test_loader, test_model.ema, epoch)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                wandb.log({"test_accuracy": top1, "epoch": epoch})
                # args.writer.add_scalar("test/acc@5", top5, args.num_eval)
                #                 wandb.log({"test/loss": test_loss,
                #                            "test/acc@1": top1,
                #                            "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                # save_checkpoint(args, {
                #     'step': step + 1,
                #     'teacher_state_dict': teacher_model.state_dict(),
                #     'student_state_dict': student_model.state_dict(),
                #     'avg_state_dict': avg_student_model.ema.state_dict() if avg_student_model is not None else None,
                #     'best_top1': args.best_top1,
                #     # 'best_top5': args.best_top5,
                #     'teacher_optimizer': t_optimizer.state_dict(),
                #     'student_optimizer': s_optimizer.state_dict(),
                #     'teacher_scheduler': t_scheduler.state_dict(),
                #     'student_scheduler': s_scheduler.state_dict(),
                #     # 'teacher_scaler': t_scaler.state_dict(),
                #     # 'student_scaler': s_scaler.state_dict(),
                # }, is_best)

    if args.local_rank in [-1, 0]:
        # args.writer.add_scalar("result/test_acc@1", args.best_top1)
        wandb.log({"result/test_acc@1": args.best_top1})
