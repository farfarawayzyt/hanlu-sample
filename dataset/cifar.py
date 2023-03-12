import logging
import math
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100, MNIST, ImageFolder
from .randaugment import RandAugmentMC
from .mydataset import ImageNet30, ImageFolder_fix
from utils.misc import search_dir_or_file
import threading

# import psutil

logger = logging.getLogger(__name__)

__all__ = ['TransformOpenMatch', 'TransformFixMatch', 'cifar10_mean',
           'cifar10_std', 'cifar100_mean', 'cifar100_std', 'normal_mean',
           'normal_std', 'TransformFixMatch_Imagenet',
           'TransformFixMatch_Imagenet_Weak']
### Enter Path of the data directory.
DATA_PATH = '/data/hanlu'

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar(args, norm=True):
    root = search_dir_or_file([os.path.join(r, args.dataset) for r in args.root])
    name = args.dataset
    if name == "cifar10":
        data_folder = datasets.CIFAR10
        data_folder_main = CIFAR10SSL
        mean = args.mean = cifar10_mean
        std = args.std = cifar10_std
        num_class = 10
    elif name == "cifar100":
        data_folder = datasets.CIFAR100
        data_folder_main = CIFAR100SSL
        mean = args.mean = cifar100_mean
        std = args.std = cifar100_std
        num_class = 100
        num_super = args.num_super

    else:
        raise NotImplementedError()
    assert num_class >= args.label_classes

    if name == "cifar10":
        base_dataset = data_folder(root, train=True, download=False)
        # args.num_classes = 6
    elif name == 'cifar100':
        base_dataset = data_folder(root, train=True,
                                   download=False)

    base_dataset.targets = np.array(base_dataset.targets)
    if name == 'cifar10':
        base_dataset.targets = (base_dataset.targets - 2) % 10
        # base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
        # base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9

    if args.vary_ratio:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs, labeled_idxs, selected_class, unselected_class = \
            x_u_split_with_ratio(args, base_dataset.targets)
    elif args.open_split:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs, labeled_idxs = x_u_split_open_match(args,
                                                                                                base_dataset.targets)
    else:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs, labeled_idxs, selected_class, unselected_class = \
            x_u_split(args, base_dataset.targets)

    unique_labeled = np.unique(labeled_idxs)
    assert len(unique_labeled) == len(labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s" % name)
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")

    ## This function will be overwritten in trainer.py
    norm_func = TransformFixMatch(mean=mean, std=std, norm=norm)
    if norm:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if name == 'cifar10':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, train=True,
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, train=True,
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, train=True,
            transform=norm_func_test)
        labeled_dataset = data_folder_main(
            root, unique_labeled, train=True,
            transform=norm_func_test)
    elif name == 'cifar100':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, train=True,
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, train=True,
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, train=True,
            transform=norm_func_test)
        labeled_dataset = data_folder_main(
            root, unique_labeled, train=True,
            transform=norm_func_test)

    if name == 'cifar10':
        train_labeled_dataset.targets = (train_labeled_dataset.targets - 2) % 10
        train_unlabeled_dataset.targets = (train_unlabeled_dataset.targets - 2) % 10
        val_dataset.targets = (val_dataset.targets - 2) % 10
        labeled_dataset.targets = (labeled_dataset.targets - 2) % 10

    if name == 'cifar10':
        test_dataset = data_folder(
            root, train=False, transform=norm_func_test, download=False)
    elif name == 'cifar100':
        test_dataset = data_folder(
            root, train=False, transform=norm_func_test,
            download=False)
    test_dataset.targets = np.array(test_dataset.targets)

    if name == 'cifar10':
        test_dataset.targets -= 2
        test_dataset.targets[np.where(test_dataset.targets == -2)[0]] = 8
        test_dataset.targets[np.where(test_dataset.targets == -1)[0]] = 9

    target_ind = np.where(test_dataset.targets >= args.label_classes)[0]
    test_dataset.targets[target_ind] = args.label_classes

    return train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset, labeled_dataset


def get_imagenet(args, norm=True):
    root = os.path.join(search_dir_or_file([os.path.join(r, 'imagenet-30') for r in args.root]), "..")
    mean = args.mean = normal_mean
    std = args.std = normal_std
    txt_labeled = "filelist/imagenet_train_labeled.txt"
    txt_unlabeled = "filelist/imagenet_train_unlabeled.txt"
    txt_val = "filelist/imagenet_val.txt"
    txt_test = "filelist/imagenet_test.txt"
    ## This function will be overwritten in trainer.py
    norm_func = TransformFixMatch_Imagenet(mean=mean, std=std,
                                           norm=norm, size_image=224)
    dataset_labeled = ImageNet30(root, txt_labeled, transform=norm_func)
    dataset_unlabeled = ImageFolder_fix(root, txt_unlabeled, transform=norm_func)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset_val = ImageNet30(root, txt_val, transform=test_transform)
    dataset_test = ImageNet30(root, txt_test, transform=test_transform)
    logger.info(f"Labeled examples: {len(dataset_labeled)}"
                f"Unlabeled examples: {len(dataset_unlabeled)}"
                f"Valdation samples: {len(dataset_val)}")
    return dataset_labeled, dataset_unlabeled, dataset_test, dataset_val, dataset_labeled


def get_tiny_imagenet(args, norm=True):
    root = search_dir_or_file([os.path.join(r, 'tiny-imagenet') for r in args.root])
    train = ImageFolder(root=os.path.join(root, 'train'))


def x_u_split(args, labels):
    classes = np.unique(labels)
    tot_classes = len(classes)
    label_per_class = args.num_labeled
    unlabel_per_class = args.n_unlabels // tot_classes
    val_per_class = args.num_val
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    if args.selected_class is None:
        args.selected_class = classes[:args.label_classes]
        unselected_class = classes[args.label_classes:]
    else:
        unselected_class = list(set(classes) - set(args.selected_class))
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in args.selected_class:
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        if label_per_class > 0:
            labeled_idx.extend(idx[:label_per_class])
            unlabeled_idx.extend(idx[:label_per_class + unlabel_per_class])
        else:
            labeled_idx.extend(idx)
            unlabeled_idx.extend(idx[:unlabel_per_class])
        val_idx.extend(idx[-val_per_class:])
    for i in unselected_class:
        idx = np.where(labels == i)[0]
        # idx = np.random.choice(idx, unlabel_per_class, False)
        np.random.shuffle(idx)
        unlabeled_idx.extend(idx[:unlabel_per_class])

    labeled_idx = np.array(labeled_idx)

    # assert len(labeled_idx) == args.num_labeled * args.label_classes
    if args.expand_labels or len(labeled_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(labeled_idx))
        train_labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    else:
        train_labeled_idx = labeled_idx.copy()
    np.random.shuffle(train_labeled_idx)
    np.random.shuffle(unlabeled_idx)
    np.random.shuffle(val_idx)
    return train_labeled_idx, np.array(unlabeled_idx), np.array(
        val_idx), labeled_idx, args.selected_class, unselected_class


def x_u_split_with_ratio(args, labels):
    # n_labels_per_cls, n_unlabels, tot_class=6, ratio=0.5, val_ratio=0.1, selected_class=None
    state = np.random.RandomState(42)
    if isinstance(labels, list):
        labels = np.array(labels)
    classes = np.unique(labels)
    # n_labels_per_cls = n_labels // tot_class

    n_unlabels_per_cls = int(args.n_unlabels * (1.0 - args.ood_ratio)) // args.label_classes
    assert (args.label_classes < len(classes))
    n_unlabels_shift = (args.n_unlabels - (n_unlabels_per_cls * args.label_classes)) // (
            len(classes) - args.label_classes)
    l_idx = []
    u_idx = []
    val_idx = []
    if args.selected_class is None:
        args.selected_class = classes[:args.label_classes]
        unselected_class = classes[args.label_classes:]
    else:
        unselected_class = list(set(classes) - set(args.selected_class))
    for c in args.selected_class:
        inds = np.where(labels == c)[0]
        state.shuffle(inds)
        # val_per_cls = int(len(inds) * args.val_ratio)
        l_idx.extend(inds[:args.num_labeled])
        assert args.num_labeled + n_unlabels_per_cls < len(inds)
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        u_idx.extend(inds[:args.num_labeled + n_unlabels_per_cls])
        val_idx.extend(inds[-args.num_val:])
    for c in unselected_class:
        inds = np.where(labels == c)[0]
        state.shuffle(inds)
        u_idx.extend(inds[:n_unlabels_shift])

    assert len(l_idx) == args.num_labeled * args.label_classes
    if args.expand_labels or len(l_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(l_idx))
        train_labeled_idx = np.hstack([l_idx for _ in range(num_expand_x)])
    else:
        train_labeled_idx = l_idx.copy()
    np.random.shuffle(train_labeled_idx)

    return train_labeled_idx, np.array(u_idx), np.array(val_idx), \
           np.array(l_idx), args.selected_class, unselected_class


# def x_u_split(args, labels):
#     # n_labels_per_cls, n_unlabels, tot_class=6, ratio=0.5, val_ratio=0.1, selected_class=None
#     state = np.random.RandomState(42)
#     if isinstance(labels, list):
#         labels = np.array(labels)
#     classes = np.unique(labels)
#     # n_labels_per_cls = n_labels // tot_class
#
#     n_unlabels_per_cls = int(args.n_unlabels * (1.0 - args.ood_ratio)) // args.label_classes
#     assert (args.label_classes < len(classes))
#     n_unlabels_shift = (args.n_unlabels - (n_unlabels_per_cls * args.label_classes)) // (
#             len(classes) - args.label_classes)
#     l_idx = []
#     u_idx = []
#     val_idx = []
#     if args.selected_class is None:
#         args.selected_class = classes[:args.label_classes]
#         unselected_class = classes[args.label_classes:]
#     else:
#         unselected_class = list(set(classes) - set(args.selected_class))
#     for c in args.selected_class:
#         inds = np.where(labels == c)[0]
#         state.shuffle(inds)
#         # val_per_cls = int(len(inds) * args.val_ratio)
#         l_idx.extend(inds[:args.num_labeled])
#         assert args.num_labeled + n_unlabels_per_cls < len(inds)
#         # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
#         u_idx.extend(inds[:args.num_labeled + n_unlabels_per_cls])
#         val_idx.extend(inds[-args.num_val:])
#     for c in unselected_class:
#         inds = np.where(labels == c)[0]
#         state.shuffle(inds)
#         u_idx.extend(inds[:n_unlabels_shift])
#
#     assert len(l_idx) == args.num_labeled * args.label_classes
#     if args.expand_labels or len(l_idx) < args.batch_size:
#         num_expand_x = math.ceil(
#             args.batch_size * args.eval_step / len(l_idx))
#         train_labeled_idx = np.hstack([l_idx for _ in range(num_expand_x)])
#     else:
#         train_labeled_idx = l_idx.copy()
#     np.random.shuffle(train_labeled_idx)
#
#     return train_labeled_idx, np.array(u_idx), np.array(val_idx), np.array(l_idx), args.selected_class, unselected_class


def x_u_split_open_match(args, labels):
    # labels, n_labels_per_cls, n_unlabels, tot_class=6, ratio=0.5, val_ratio=0.1, selected_class=None
    label_per_class = args.num_labeled  # // args.num_classes
    val_per_class = args.num_val  # // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class + val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.expand_labels or len(labeled_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(labeled_idx))
        train_labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    else:
        train_labeled_idx = labeled_idx.copy()
    np.random.shuffle(train_labeled_idx)

    # if not args.no_out:
    unlabeled_idx = np.array(range(len(labels)))
    # unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
    return train_labeled_idx, unlabeled_idx, val_idx, labeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        # self.weak2 = transforms.Compose([
        #     transforms.RandomHorizontalFlip(), ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong)
        else:
            return weak, strong


class TransformContrast(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(), ])
        # self.strong = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(size=size_image,
        #                           padding=int(size_image * 0.125),
        #                           padding_mode='reflect'),
        #     RandAugmentMC(n=2, m=10)]
        # )
        s = 0.5
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size_image, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(), ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformFixMatch_Imagenet(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong)
        else:
            return weak, strong


class TransformFixMatch_Imagenet_Weak(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        # if indexs is not None:
        #     self.data = self.data[indexs]
        #     self.targets = np.array(self.targets)[indexs]
        self.targets = np.array(self.targets)
        self.initial_index = self.indexes = indexs
        self.return_idx = return_idx
        # self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.indexes = self.initial_index[indexes]

    def init_index(self):
        self.indexes = self.initial_index

    def __getitem__(self, index):

        # t = threading.currentThread()
        # print(f"pid {os.getpid()}, thead id {t.ident}, name {t.getName()}")
        img, target = self.data[self.indexes[index]], self.targets[self.indexes[index]]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.indexes)


# class CIFAR100FIX(datasets.CIFAR100):
#     def __init__(self, root, num_super=10, train=True, transform=None,
#                  target_transform=None, download=False, return_idx=False):
#         super().__init__(root, train=train, transform=transform,
#                          target_transform=target_transform, download=download)
#
#         coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
#                                   3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
#                                   6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
#                                   0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
#                                   5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
#                                   16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
#                                   10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
#                                   2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
#                                   16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
#                                   18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
#         self.course_labels = coarse_labels[self.targets]
#         self.targets = np.array(self.targets)
#         labels_unknown = self.targets[np.where(self.course_labels > num_super)[0]]
#         labels_known = self.targets[np.where(self.course_labels <= num_super)[0]]
#         unknown_categories = np.unique(labels_unknown)
#         known_categories = np.unique(labels_known)
#
#         num_unknown = len(unknown_categories)
#         num_known = len(known_categories)
#         print("number of unknown categories %s" % num_unknown)
#         print("number of known categories %s" % num_known)
#         assert num_known + num_unknown == 100
#         # new_category_labels = list(range(num_known))
#         self.targets_new = np.zeros_like(self.targets)
#         for i, known in enumerate(known_categories):
#             ind_known = np.where(self.targets == known)[0]
#             self.targets_new[ind_known] = i
#         for i, unknown in enumerate(unknown_categories):
#             ind_unknown = np.where(self.targets == unknown)[0]
#             self.targets_new[ind_unknown] = num_known + i
#
#         self.targets = self.targets_new
#         assert len(np.where(self.targets >= num_known)[0]) == len(labels_unknown)
#         assert len(np.where(self.targets < num_known)[0]) == len(labels_known)
#         self.num_known_class = num_known
#
#     def __getitem__(self, index):
#
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = np.array(self.targets)
        self.initial_index = self.indexes = indexs
        self.return_idx = return_idx
        # self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.indexes = self.initial_index[indexes]

    def init_index(self):
        self.indexes = self.initial_index

    # def set_index(self, indexes=None):
    #     if indexes is not None:
    #         self.data_index = self.data[indexes]
    #         self.targets_index = self.targets[indexes]
    #     else:
    #         self.data_index = self.data
    #         self.targets_index = self.targets
    #
    # def init_index(self):
    #     self.data_index = self.data
    #     self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data[self.indexes[index]], self.targets[self.indexes[index]]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.indexes)


def get_transform(mean, std, image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_ood(dataset, id, test_only=False, image_size=None):
    image_size = (32, 32, 3) if image_size is None else image_size
    if id == "cifar10":
        mean = cifar10_mean
        std = cifar10_std
    elif id == "cifar100":
        mean = cifar100_mean
        std = cifar100_std
    elif "imagenet" in id or id == "tiny":
        mean = normal_mean
        std = normal_std

    _, test_transform = get_transform(mean, std, image_size=image_size)

    if dataset == 'cifar10':
        test_set = datasets.CIFAR10(os.path.join(DATA_PATH, 'cifar10'), train=False, download=False,
                                    transform=test_transform)

    elif dataset == 'cifar100':
        test_set = datasets.CIFAR100(os.path.join(DATA_PATH, 'cifar100'), train=False, download=False,
                                     transform=test_transform)

    elif dataset == 'svhn':
        test_set = datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='test', download=False,
                                 transform=test_transform)

    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'ood_detect', 'LSUN_pil')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'ood_detect', 'Imagenet_FIX')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'ood_detect', 'dogs', 'Images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'cub')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'flowers102':
        test_dir = os.path.join(DATA_PATH, 'ssl_transfer/flowers_new/train')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'food_101':
        test_dir = os.path.join(DATA_PATH, 'ssl_transfer', 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, 'ssl_transfer/caltech101/101_ObjectCategories')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'dtd':
        test_dir = os.path.join(DATA_PATH, 'ssl_transfer', 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'pets':
        test_dir = os.path.join(DATA_PATH, 'ssl_transfer', 'pets', "train")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    return test_set


DATASET_GETTERS = {'cifar10': get_cifar,
                   'cifar100': get_cifar,
                   'imagenet': get_imagenet,
                   'tiny-imagenet': get_imagenet,
                   }
