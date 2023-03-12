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
from utils import AverageMeter, save_checkpoint, create_model, accuracy, exclude_ood, roc_id_ood
from utils.tsne import tsne
from sklearn.metrics import roc_auc_score
# from lightly.loss import ntx_ent_loss, DINOLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SimCLRProjectionHead
from torchvision.transforms.functional import to_pil_image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, sims, mask=None, neg_mask=None, reduction="mean"):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            sims: similarity matrix of shape [bsz,bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = sims.device
        bsz = sims.size(0)
        # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)

        # for numerical stability
        sims = sims / self.base_temperature
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        if neg_mask is None:
            neg_mask = torch.ones_like(mask)
        # mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = -  mean_log_prob_pos

        if reduction == "mean":
            loss = loss.mean()

        return loss


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    unk_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            pred = torch.sum(F.softmax(outputs, dim=1)[:, args.label_classes:], dim=1)
            unk_all.append(pred)
            batch_time.update(time.time() - end)
            end = time.time()
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = torch.cat(unk_all)
    unk_all = unk_all.data.cpu().numpy()
    roc = roc_id_ood(test_id, unk_all)

    return roc


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    last = (epoch == (args.epochs - 1))

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    if last:
        mean = torch.as_tensor(args.mean, dtype=torch.float32, device=args.device).view(-1, 1, 1)
        std = torch.as_tensor(args.std, dtype=torch.float32, device=args.device).view(-1, 1, 1)
        columns = ["image", "truth", "guess"]
        for digit in range(args.label_classes):
            columns.append("score_" + str(digit))
        table_data = wandb.Table(columns=columns)
        logger.info('Last epoch: enable logging wandb tables')

    pred_list = []
    target_list = []
    emb_list = []
    known_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            targets = targets.to(args.device)
            known_targets = targets < args.label_classes  # [0]
            unkown_targets = targets >= args.label_classes
            inputs = inputs.to(args.device)
            outputs, embs = model(inputs, emb=True)
            emb_list.append(embs)
            pred = F.softmax(outputs, dim=1)
            pred_list.append(pred)
            target_list.append(targets)
            known_all.append(torch.sum(pred[known_targets, args.label_classes:], dim=1))
            outputs = outputs[:, : args.label_classes]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            loss = F.cross_entropy(known_pred, known_targets)
            if last:
                for im, lab, pred in zip(inputs, known_targets, known_pred):
                    img = wandb.Image(to_pil_image(im.mul_(std).add_(mean)))
                    pred_label = torch.argmax(pred).item()
                    table_data.add_data(img, lab, pred_label, *F.softmax(pred, dim=0).cpu())
            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                losses.update(loss.item(), known_pred.shape[0])
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()
    preds = torch.cat(pred_list)
    targets = torch.cat(target_list)
    known_all = torch.cat(known_all)

    ood_label = (targets >= args.label_classes).long().cpu().numpy()
    confidence_score = 1 - torch.max(preds[:, :args.label_classes], dim=1)[0].cpu().numpy()
    classifier_score = torch.sum(preds[:, args.label_classes:], dim=1).cpu().numpy()
    confidence_auc = roc_auc_score(ood_label, confidence_score)
    classifier_auc = roc_auc_score(ood_label, classifier_score)
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))

    ret_dict = {}

    if last:
        test_data_at = wandb.Artifact(wandb.run.name.replace('@', '-') + '_' + str(wandb.run.id), type="model")
        cm = plt.get_cmap('tab10')
        test_embs = torch.cat(emb_list)
        tsne_emb = tsne(test_embs, verbose=True, device=test_embs.device).cpu().numpy()
        fig = plt.figure()
        labels = targets.cpu().numpy()
        ood_idx = labels >= args.label_classes
        plt.scatter(tsne_emb[ood_idx, 0], tsne_emb[ood_idx, 1], 5, c='grey')
        plt.scatter(tsne_emb[~ood_idx, 0], tsne_emb[~ood_idx, 1], 5, labels[~ood_idx], cmap=cm)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        embdding_file = "images/embed" + '_' + str(wandb.run.id) + ".pdf"
        plt.savefig(embdding_file, format='pdf', dpi=800, transparent=True,
                    bbox_inches='tight')
        test_data_at.add_file(embdding_file, "embedding.pdf")
        wandb.log({"embeddings": wandb.Image(plt)})
        plt.close(fig)
        if args.save_model:
            model_file = "checkpoint/" + str(wandb.run.id) + ".pth"
            torch.save(model.state_dict(), model_file)
            test_data_at.add_file(embdding_file, "model.pt")
        test_data_at.add(table_data, "predictions")
        wandb.run.log_artifact(test_data_at)
    ret_dict["known_all"] = known_all.cpu().numpy()
    ret_dict.update({"test/2.test_loss": losses.avg, "test_accuracy": top1.avg, "detect/confidence_auc": confidence_auc,
                     "detect/classifier_auc": classifier_auc})
    return ret_dict


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


def soft_ce(q, z, temperature=1.0):
    return - torch.mean(
        torch.sum(q * F.log_softmax(z / temperature, dim=1), dim=1)
    )


def bc_coefficient(p, q):
    return torch.sqrt(p) @ torch.sqrt(q).T


# def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
#           model, optimizer, ema_model, scheduler):
def train(args, train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler, labeled_dataloader):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

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
    # contrast_loss = DINOLoss(args.label_classes, warmup_teacher_temp=1., teacher_temp=1., student_temp=1.0).to(
    #     args.device)
    # contrast_loss = DINOLoss(args.label_classes).to(
    #     args.device)
    eps = 1e-6
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        mask_acc = AverageMeter()
        ood_portion = AverageMeter()
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
            mask = mask.float()

            pos = - F.cross_entropy(logits_u_s, targets_u, reduction='none').unsqueeze(1)
            neg = F.log_softmax(logits_u_s, dim=1) @ F.softmax(logits_u_s, dim=1).T
            logits = torch.cat([pos, neg], dim=1)
            labels = torch.zeros(logits.size(0), device=logits.device).long()
            Lu = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
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
            mask_acc.update((((targets_u == label_u).float() * mask).sum() / (mask.sum() + 1e-6)).item())
            if args.dummy_classes > 0:
                ood_probs = torch.sum(pseudo_label[:, args.label_classes:], dim=1)
                ood_portion.update(ood_probs.mean().item())
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
            out_dict = test(args, test_loader, test_model, epoch)
            for k, v in out_dict.items():
                if isinstance(v, float):
                    wandb.log({k: v, 'epoch': epoch})
            if args.test_ood and epoch % 20 == 0:
                for ood in ood_loaders.keys():
                    roc_ood = test_ood(args, out_dict["known_all"], ood_loaders[ood], test_model)
                    wandb.log({f"detect/{ood}_roc": roc_ood, 'epoch': epoch})
                    logger.info("ROC vs {ood}: {roc}".format(ood=ood, roc=roc_ood))

            wandb.log({'train_loss': losses.avg, 'epoch': epoch})
            wandb.log({'train/2.train_loss_x': losses_x.avg, 'epoch': epoch})
            wandb.log({'train/3.train_loss_u': losses_u.avg, 'epoch': epoch})
            wandb.log({'train/4.mask': mask_probs.avg, 'epoch': epoch})
            wandb.log({'train/5.mask_acc': mask_acc.avg, 'epoch': epoch})
            wandb.log({'train/6.ood_portion': ood_portion.avg, 'epoch': epoch})
            # wandb.log({'test_accuracy': test_acc, 'epoch': epoch})
            # wandb.log({'test/2.test_loss': test_loss, 'epoch': epoch})
            test_acc = out_dict["test_accuracy"]
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
