import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
# import wandb
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import glob
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from utils import AverageMeter, save_checkpoint, create_model, accuracy, exclude_ood, test

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
        # from torch.cuda import amp
        scaler = torch.cuda.amp.GradScaler()
        from torch.cuda.amp import autocast
    global best_acc
    test_accs = []
    end = time.time()

    # if args.dataset == 'cifar10':
    #     mean = cifar10_mean
    #     std = cifar10_std
    #     func_trans = TransformOpenMatch
    # elif args.dataset == 'cifar100':
    #     mean = cifar100_mean
    #     std = cifar100_std
    #     func_trans = TransformOpenMatch
    # elif 'imagenet' in args.dataset:
    #     mean = normal_mean
    #     std = normal_std
    #     func_trans = TransformFixMatch_Imagenet_Weak

    if args.exclude_ood:
        ## pick pseudo-inliers
        assert args.ood_ratio > 0
        exclude_ood(args, unlabeled_dataset)

    # labeled_dataset = copy.deepcopy(train_labeled_trainloader.dataset)
    # labeled_dataset.transform = func_trans(mean=mean, std=std)
    # train_labeled_trainloader = DataLoader(
    #     labeled_dataset,
    #     sampler=train_sampler(labeled_dataset),
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     drop_last=True)
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
    # run_id = str(wandb.run.id)
    # profile_dir = f'./log/{run_id}'
    artifact = None
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    for epoch in range(args.start_epoch, args.epochs):

        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        mask_acc = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.eval_step):
            try:
                # (inputs_x, _), targets_x = labeled_iter.next()
                # error occurs ↓
                (inputs_x, _), targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_labeled_trainloader)
                # (inputs_x, _), targets_x = labeled_iter.next()
                # error occurs ↓
                (inputs_x, _), targets_x = next(labeled_iter)

            try:
                # (inputs_u_w, inputs_u_s), label_u = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), label_u = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), label_u = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), label_u = next(unlabeled_iter)
            data_time_ = time.time() - end
            data_time.update(time.time() - end)
            data_start = time.time()
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            to_time_ = time.time() - data_start
            forward_start = time.time()
            if args.amp:
                with autocast():
                    Lu, Lx, loss, mask, targets_u = compute_loss(args, batch_size, inputs, label_u, model,
                                                                 targets_x)
            else:
                Lu, Lx, loss, mask, targets_u = compute_loss(args, batch_size, inputs, label_u, model, targets_x)
            forward_time = time.time() - forward_start
            backward_start = time.time()
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
            backward_time = time.time() - backward_start

            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time_ = time.time() - end
            batch_time.update(batch_time_)
            mask_probs.update(mask.mean())
            mask_acc.update((((targets_u == label_u.to(args.device)).float() * mask).sum() / (mask.sum() + 1e-6)))
            if not args.no_progress:
                # p_bar.set_description(f"Time cost: Data: {data_time_:.4f}, To: {to_time_:.4f}, Batch: {batch_time_:.4f}")
                print(f"Time cost: Data: {data_time_:.4f}, To: {to_time_:.4f}, Forward: {forward_time:.4f}, Backward: {backward_time:.4f}, Total: {batch_time_:.4f}")
                # p_bar.set_description(
                #     "Train Epoch: {epoch}/{epochs:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                #         epoch=epoch + 1,
                #         epochs=args.epochs,
                #         lr=scheduler.get_last_lr()[0],
                #         data=data_time.avg,
                #         bt=batch_time.avg,
                #         loss=losses.avg,
                #         loss_x=losses_x.avg,
                #         loss_u=losses_u.avg,
                #         mask=mask_probs.avg))
                p_bar.update()
            end = time.time()
            # prof.step()

        if not args.no_progress:
            p_bar.close()

        # if args.local_rank in [-1, 0]:
            # out_dict = test(args, test_loader, test_model, epoch)
            # for k, v in out_dict.items():
            #     if isinstance(v, float):
            #         wandb.log({k: v, 'epoch': epoch})
            # test_acc = out_dict["test_accuracy"]
            # wandb.log({'train_loss': losses.avg, 'epoch': epoch})
            # wandb.log({'train/2.train_loss_x': losses_x.avg, 'epoch': epoch})
            # wandb.log({'train/3.train_loss_u': losses_u.avg, 'epoch': epoch})
            # wandb.log({'train/4.mask': mask_probs.avg, 'epoch': epoch})
            # wandb.log({'train/5.mask_acc': mask_acc.avg, 'epoch': epoch})
            # wandb.log({'train/6.batch_time': batch_time.avg, 'epoch': epoch})
            # wandb.log({'train/7.data_time': data_time.avg, 'epoch': epoch})
            # args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            # wandb.log({'test_accuracy': test_acc, 'epoch': epoch})
            # is_best = test_acc > best_acc
            # best_acc = max(test_acc, best_acc)

            # if "artifact" in out_dict:
            #     artifact = out_dict["artifact"]

            #     if args.cRT:
            #         # scd_acc = AverageMeter()
            #         # model.eval()
            #         classifier_retrain(args, epoch, labeled_dataloader, test_loader, test_model)

                # if args.save_model:
                #     model_file = "checkpoint/" + str(wandb.run.id) + ".pth"
                #     torch.save(test_model.state_dict(), model_file)
                #     artifact.add_file(model_file, "model.pth")

                # model_to_save = model.module if hasattr(model, "module") else model
                # if args.use_ema:
                #     model_to_save = ema_model.ema.module if hasattr(
                #         ema_model.ema, "module") else ema_model.ema

            # test_accs.append(test_acc)
            # logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            # logger.info('Mean top-1 acc: {:.2f}\n'.format(
            #     np.mean(test_accs[-20:])))
            # if "artifact" in out_dict:
            #     artifact = out_dict["artifact"]
    # profile_artifact = wandb.Artifact(wandb.run.name.replace('@', '-') + '_' + str(wandb.run.id), type="profile")
    # if artifact:
    #     print("log artifact")
    #     artifact.add_file(glob.glob(f"{profile_dir}/*.pt.trace.json")[0], "trace.pt.trace.json")
    #     wandb.run.log_artifact(artifact)
    # if args.local_rank in [-1, 0]:
    #     args.writer.close()


def classifier_retrain(args, epoch, labeled_dataloader, test_loader, test_model):
    train_embs, test_embs = [], []
    train_labels, test_labels = [], []
    with torch.no_grad():
        for train_x, targets_x in tqdm(labeled_dataloader, total=len(labeled_dataloader)):
            train_embs.append(test_model(train_x.to(args.device), emb=True)[1])
            train_labels.append(targets_x.to(args.device))
        for test_x, test_y in tqdm(test_loader, total=len(test_loader)):
            test_y = test_y.to(args.device)
            known_targets = test_y < args.num_classes  # [0]
            known_embs = test_model(test_x.to(args.device)[known_targets], emb=True)[1]
            known_targets = test_y[known_targets]
            test_embs.append(known_embs)
            test_labels.append(known_targets)
    train_embs = torch.cat(train_embs)
    train_labels = torch.cat(train_labels)
    test_embs = torch.cat(test_embs)
    test_labels = torch.cat(test_labels)
    train_linear_clf(train_embs, train_labels, test_embs, test_labels)
    # test_co_logits = clf(test_embs)
    # co_acc = count_acc(test_co_logits, test_labels)
    # joint_acc = count_acc(test_model.fc(test_embs), test_labels)
    # train_co_logits = clf(train_embs)
    # train_co_acc = count_acc(train_co_logits, train_labels)
    # logger.info(f'correcter accuracy : {co_acc}')
    # logger.info(f'joint accuracy : {joint_acc}')
    # # logger.info(f'correcter train accuracy : {train_co_acc}')
    # wandb.log({'correcter accuracy': co_acc, 'epoch': epoch})
    # wandb.log({'joint accuracy': joint_acc, 'epoch': epoch})


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


def train_linear_clf(x_train, y_train, x_test, y_test, epoch=500):
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
        for e in pbar:
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
            with torch.no_grad():
                test_logits = clf(x_test)
                test_acc = count_acc(test_logits, y_test)
                # wandb.log({"retrain/rt_test_accuracy": test_acc, "epoch": e})
                train_logits = clf(x_train)
                train_acc = count_acc(train_logits, y_train)
                # wandb.log({"retrain/rt_train_accuracy": train_acc, "epoch": e})
    clf.eval()
    return clf
