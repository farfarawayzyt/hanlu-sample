screen -dmS dum1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --dummy_classes 4'
#screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --dummy_classes 4'
#screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS dum5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --dummy_classes 4'

screen -dmS dum1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.05 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.5 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 1.0 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'


screen -dmS dum1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/dum --arch wideresnet --lambda_h 0.1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'


# CIFAR100

screen -dmS dum1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS dum5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

# exclude ood experiment
screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 --exclude_ood'
screen -dmS dum5 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --exclude_ood'


screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_dummy_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS dum5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model oracle_dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

screen -dmS dum2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_dummy_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS dum3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_dummy_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
screen -dmS dum4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_dummy_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS dum5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/dum --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model os_dummy_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

