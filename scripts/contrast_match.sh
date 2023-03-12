screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS con5 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'


# CIFAR100

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model contrast_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
