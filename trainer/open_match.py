import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, ova_loss, \
    save_checkpoint, ova_ent, \
    test_ood, exclude_dataset, accuracy_open, compute_roc, accuracy
import wandb

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


def test(args, test_loader, model, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_score = outputs.max(1)[0]
            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1))  # [0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1)))
            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    # import pdb
    # pdb.set_trace()
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc = compute_roc(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        ind_known = np.where(label_all < int(outputs.size(1)))[0]
        id_score = unk_all[ind_known]
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        logger.info("Overall acc: {:.3f}".format(acc.avg))
        logger.info("Unk acc: {:.3f}".format(unk.avg))
        logger.info("ROC: {:.3f}".format(roc))
        logger.info("ROC Softmax: {:.3f}".format(roc_soft))
        return {"test/2.test_loss": losses.avg, "test_accuracy": top1.avg, "detect/confidence_auc": roc,
                "known_all": id_score}
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler, labeled_dataloader):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"
    output_args = vars(args)

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
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

    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    for epoch in range(args.start_epoch, args.epochs):
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            exclude_dataset(args, unlabeled_dataset, ema_model.ema)

        unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler=train_sampler(unlabeled_dataset),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                               sampler=train_sampler(unlabeled_dataset_all),
                                               batch_size=args.batch_size * args.mu,
                                               num_workers=args.num_workers,
                                               drop_last=True)

        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        for batch_idx in range(args.eval_step):
            ## Data loading

            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s), _ = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s), _ = unlabeled_all_iter.next()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
            inputs = torch.cat([inputs_x, inputs_x_s,
                                inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)
            ## Feed data
            logits, logits_open = model(inputs)
            logits_open_u1, logits_open_u2 = logits_open[2 * b_size:].chunk(2)

            ## Loss for labeled samples
            Lx = F.cross_entropy(logits[:2 * b_size],
                                 targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(logits_open[:2 * b_size], targets_x.repeat(2))

            ## Open-set entropy minimization
            L_oem = ova_ent(logits_open_u1) / 2.
            L_oem += ova_ent(logits_open_u2) / 2.

            ## Soft consistenty regularization
            logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
            logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
            logits_open_u1 = F.softmax(logits_open_u1, 1)
            logits_open_u2 = F.softmax(logits_open_u2, 1)
            L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                logits_open_u1 - logits_open_u2) ** 2, 1), 1))

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                logits, logits_open_fix = model(inputs_ws)
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())

            else:
                L_fix = torch.zeros(1).to(args.device).mean()
            loss = Lx + Lo + args.lambda_oem * L_oem \
                   + args.lambda_socr * L_socr + L_fix
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)
            out_dict = test(args, test_loader, test_model, epoch)
            for k, v in out_dict.items():
                if isinstance(v, float):
                    wandb.log({k: v, 'epoch': epoch})
            # for ood in ood_loaders.keys():
            #     roc_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
            #     logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))
            test_acc_close = out_dict["test_accuracy"]
            wandb.log({'train_loss': losses.avg, 'epoch': epoch})
            wandb.log({'train/2.train_loss_x': losses_x.avg, 'epoch': epoch})
            wandb.log({'train/3.train_loss_o': losses_o.avg, 'epoch': epoch})
            wandb.log({'train/4.train_loss_oem': losses_oem.avg, 'epoch': epoch})
            wandb.log({'train/5.train_loss_socr': losses_socr.avg, 'epoch': epoch})
            wandb.log({'train/5.train_loss_fix': losses_fix.avg, 'epoch': epoch})
            wandb.log({'train/6.mask': mask_probs.avg, 'epoch': epoch})
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            # wandb.log({'test_accuracy': test_acc_close, 'epoch': epoch})
            # wandb.log({'test/2.test_loss': test_loss, 'epoch': epoch})

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                # overall_valid = test_overall
                close_valid = test_acc_close
                # unk_valid = test_unk
                # roc_valid = test_roc
                # roc_softm_valid = test_roc_softm
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                # 'acc close': test_acc_close,
                # 'acc overall': test_overall,
                # 'unk': test_unk,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            # logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            # logger.info('Valid overall acc: {:.3f}'.format(overall_valid))
            # logger.info('Valid unk acc: {:.3f}'.format(unk_valid))
            # logger.info('Valid roc: {:.3f}'.format(roc_valid))
            # logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
    if args.local_rank in [-1, 0]:
        args.writer.close()
