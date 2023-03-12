'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time

# import wandb
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
from torchvision.transforms.functional import to_pil_image, normalize
import os

from utils.tsne import tsne
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter',
           'accuracy_open', 'ova_loss', 'compute_roc',
           'roc_id_ood', 'ova_ent', 'exclude_dataset',
           'test_ood', 'test', 'exclude_ood', "MemoryBankModule", "search_dir_or_file"]


def search_dir_or_file(dirs, description='data directory'):
    found = None

    for d in dirs:
        print(f'searching : {d}')
        if os.path.exists(d):
            found = d
            break
    if found is None:
        raise FileNotFoundError(f'{description} not found')
    print(f'{description} : {found}')
    return found


class MemoryBankModule(torch.nn.Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.


    """

    def __init__(self, size: int = 2 ** 16):

        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f'Illegal memory bank size {size}, must be non-negative.'
            raise ValueError(msg)

        self.size = size
        self.register_buffer("bank", tensor=torch.empty(0, dtype=torch.float), persistent=False)
        self.register_buffer("bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        self.bank = torch.randn(dim, self.size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[:self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr:ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(self,
                output: torch.Tensor,
                labels: torch.Tensor = None,
                update: bool = False):
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank


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
        # table_data = wandb.Table(columns=columns)
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
                for im, lab, p in zip(inputs, targets, outputs):
                    img = wandb.Image(to_pil_image(im.mul_(std).add_(mean)))
                    pred_label = torch.argmax(p).item()
                    table_data.add_data(img, lab, pred_label, *F.softmax(p, dim=0).cpu())
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

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))

    ood_label = (targets >= args.label_classes).long().cpu().numpy()
    confidence_score = torch.max(preds[:, :args.label_classes], dim=1)[0].cpu().numpy()

    confidence_auc = roc_auc_score(ood_label, 1 - confidence_score)
    return_dict = {"test/2.test_loss": losses.avg, "test_accuracy": top1.avg, "detect/confidence_auc": confidence_auc,
                   }
    if last:
        labels = targets.cpu().numpy()
        ood_idx = labels >= args.label_classes
        known_conf = confidence_score[~ood_idx]
        unknown_conf = confidence_score[ood_idx]
        known_conf_file = "results/known_conf" + '_' + str(wandb.run.id) + ".csv"
        np.savetxt(known_conf_file, known_conf, delimiter=",")
        unknown_conf_file = "results/unknown_conf" + '_' + str(wandb.run.id) + ".csv"
        np.savetxt(unknown_conf_file, unknown_conf, delimiter=",")
        test_data_at = wandb.Artifact(wandb.run.name.replace('@', '-') + '_' + str(wandb.run.id), type="model")
        test_data_at.add_file(known_conf_file, "known_conf.csv")
        test_data_at.add_file(unknown_conf_file, "unknown_conf.csv")

        # cm = plt.get_cmap('tab10')
        # test_embs = torch.cat(emb_list)
        # tsne_emb = tsne(test_embs, verbose=True, device=test_embs.device).cpu().numpy()
        # fig = plt.figure()
        #
        # plt.scatter(tsne_emb[ood_idx, 0], tsne_emb[ood_idx, 1], 5, c='grey')
        # plt.scatter(tsne_emb[~ood_idx, 0], tsne_emb[~ood_idx, 1], 5, labels[~ood_idx], cmap=cm)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        #
        # embdding_file = "images/embed" + '_' + str(wandb.run.id) + ".pdf"
        # plt.savefig(embdding_file, format='pdf', dpi=800, transparent=True,
        #             bbox_inches='tight')
        # test_data_at.add_file(embdding_file, "embedding.pdf")
        # wandb.log({"embeddings": wandb.Image(plt)})
        # plt.close(fig)

        # if args.save_model:
        #     model_file = "checkpoint/" + str(wandb.run.id) + ".pth"
        #     torch.save(model.state_dict(), model_file)
        #     test_data_at.add_file(model_file, "model.pth")
        test_data_at.add(table_data, "predictions")
        # wandb.run.log_artifact(test_data_at)
        return_dict["artifact"] = test_data_at
    return return_dict


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / (batch_size + 1e-18)))
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def roc_id_ood(score_id, score_ood):
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0] + score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all)


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                         torch.log(logits_open + 1e-8), 1), 1))
    return Le


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def exclude_ood(args, dataset):
    dataset.init_index()
    targets = dataset.targets[dataset.indexes]
    ind_selected = np.where(targets < args.label_classes)[0]
    print("selected ratio %s" % (len(ind_selected) / len(targets)))
    # ratio = args.label_classes * args.num_labeled / len(targets)
    # print(f'theoretical ratio {ratio}')
    dataset.set_index(ind_selected)


def exclude_dataset(args, dataset, model, exclude_known=False):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
        if not args.no_progress:
            test_loader.close()
    known_all = known_all.data.cpu().numpy()
    if exclude_known:
        ind_selected = np.where(known_all == 0)[0]
    else:
        ind_selected = np.where(known_all != 0)[0]
    print("selected ratio %s" % ((len(ind_selected) / len(known_all))))
    model.train()
    dataset.set_index(ind_selected)


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    roc = roc_id_ood(test_id, unk_all)

    return roc

# def test(args, test_loader, model, epoch, val=False):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     acc = AverageMeter()
#     unk = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     if not args.no_progress:
#         test_loader = tqdm(test_loader,
#                            disable=args.local_rank not in [-1, 0])
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             data_time.update(time.time() - end)
#             model.eval()
#             inputs = inputs.to(args.device)
#             targets = targets.to(args.device)
#             outputs, outputs_open = model(inputs)
#             outputs = F.softmax(outputs, 1)
#             out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
#             tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
#             pred_close = outputs.data.max(1)[1]
#             unk_score = out_open[tmp_range, 0, pred_close]
#             known_score = outputs.max(1)[0]
#             targets_unk = targets >= int(outputs.size(1))
#             targets[targets_unk] = int(outputs.size(1))
#             known_targets = targets < int(outputs.size(1))#[0]
#             known_pred = outputs[known_targets]
#             known_targets = targets[known_targets]
#
#             if len(known_pred) > 0:
#                 prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
#                 top1.update(prec1.item(), known_pred.shape[0])
#                 top5.update(prec5.item(), known_pred.shape[0])
#
#             ind_unk = unk_score > 0.5
#             pred_close[ind_unk] = int(outputs.size(1))
#             acc_all, unk_acc, size_unk = accuracy_open(pred_close,
#                                                        targets,
#                                                        num_classes=int(outputs.size(1)))
#             acc.update(acc_all.item(), inputs.shape[0])
#             unk.update(unk_acc, size_unk)
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#             if batch_idx == 0:
#                 unk_all = unk_score
#                 known_all = known_score
#                 label_all = targets
#             else:
#                 unk_all = torch.cat([unk_all, unk_score], 0)
#                 known_all = torch.cat([known_all, known_score], 0)
#                 label_all = torch.cat([label_all, targets], 0)
#
#             if not args.no_progress:
#                 test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
#                                             "Data: {data:.3f}s."
#                                             "Batch: {bt:.3f}s. "
#                                             "Loss: {loss:.4f}. "
#                                             "Closed t1: {top1:.3f} "
#                                             "t5: {top5:.3f} "
#                                             "acc: {acc:.3f}. "
#                                             "unk: {unk:.3f}. ".format(
#                     batch=batch_idx + 1,
#                     iter=len(test_loader),
#                     data=data_time.avg,
#                     bt=batch_time.avg,
#                     loss=losses.avg,
#                     top1=top1.avg,
#                     top5=top5.avg,
#                     acc=acc.avg,
#                     unk=unk.avg,
#                 ))
#         if not args.no_progress:
#             test_loader.close()
#     ## ROC calculation
#     #import pdb
#     #pdb.set_trace()
#     unk_all = unk_all.data.cpu().numpy()
#     known_all = known_all.data.cpu().numpy()
#     label_all = label_all.data.cpu().numpy()
#     if not val:
#         roc = compute_roc(unk_all, label_all,
#                           num_known=int(outputs.size(1)))
#         roc_soft = compute_roc(-known_all, label_all,
#                                num_known=int(outputs.size(1)))
#         ind_known = np.where(label_all < int(outputs.size(1)))[0]
#         id_score = unk_all[ind_known]
#         logger.info("Closed acc: {:.3f}".format(top1.avg))
#         logger.info("Overall acc: {:.3f}".format(acc.avg))
#         logger.info("Unk acc: {:.3f}".format(unk.avg))
#         logger.info("ROC: {:.3f}".format(roc))
#         logger.info("ROC Softmax: {:.3f}".format(roc_soft))
#         return losses.avg, top1.avg, acc.avg, \
#                unk.avg, roc, roc_soft, id_score
#     else:
#         logger.info("Closed acc: {:.3f}".format(top1.avg))
#         return top1.avg
