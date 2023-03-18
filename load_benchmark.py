import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from utils import set_parser, set_dataset

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class EasyTimer:
    def __init__(self) -> None:
        self.before = time.time()

    def get_and_update(self):
        cur = time.time()
        ret = cur - self.before
        self.before = cur 
        return ret


if __name__ == '__main__':
    args = set_parser()
    # print(args)
    args.device = torch.device('cuda', args.gpu_id)
    args.root = [args.root]
    args.ood_data = ['lsun', 'dtd', 'cub', 'flowers102', 'caltech_256', 'stanford_dogs']
    args.image_size = (224, 224, 3)

    train_labeled_trainloader, unlabeled_dataset, test_loader, val_loader, ood_loaders, labeled_loader \
        = set_dataset(args)
    
    unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                    sampler=RandomSampler(unlabeled_dataset),
                                    batch_size=args.batch_size * args.mu,
                                    num_workers=args.num_workers,
                                    drop_last=True,
                                    pin_memory=True)

    labeled_iter = iter(train_labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    # time_point = time.time()
    timer = EasyTimer()
    for __ in range(1):
        for batch_idx in range(args.eval_step):
            try:
                # (inputs_x, _), targets_x = labeled_iter.next()
                # error occurs ↓
                (inputs_x, _), targets_x = next(labeled_iter)
            except Exception as ex:
                # print(ex)
                # raise
                labeled_iter = iter(train_labeled_trainloader)
                # (inputs_x, _), targets_x = labeled_iter.next()
                # error occurs ↓
                (inputs_x, _), targets_x = next(labeled_iter)

            print(inputs_x[0].shape)

            try:
                # (inputs_u_w, inputs_u_s), label_u = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), label_u = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), label_u = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), label_u = next(unlabeled_iter)

            load_time = timer.get_and_update()

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            
            to_time = timer.get_and_update()

            # time.sleep(0.0085)
            inputs += 1

            dummy_time = timer.get_and_update()
            print(f'Load: {load_time:.4f}, To: {to_time:.4f}')
