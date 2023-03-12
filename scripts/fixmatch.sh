screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200  \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200  \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.95 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 25 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.9 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

# exclude ood experiment
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --exclude_ood'


screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

# CIFAR100

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

# exclude ood experiment
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --exclude_ood'


screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'


screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'
screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'

screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --no_test_ood'


# very scarced data

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 4 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 10 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 25 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 4 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.9 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 10 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.9 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 25 --out checkpoint/fix --arch wideresnet --epochs 150 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.9 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood'


screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'


screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 5 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 10 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 25 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --save_model \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 7 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --vary_ratio --no_test_ood --cRT'