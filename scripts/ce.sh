screen -dmS vce bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/ce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS osce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS fix bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS om bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model open_match --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 \
--lambda_oem 0.1 --lambda_socr 0.5 ; exec bash'

screen -dmS vce bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/ce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --strong_aug'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --strong_aug'
screen -dmS osce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --strong_aug'

screen -dmS vce bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/ce --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/osce --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS osce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/orce --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

screen -dmS vce bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/ce --arch resnet18 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch resnet18 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'
screen -dmS osce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch resnet18 \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

# cifar10

screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .5 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .75 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug'

screen -dmS om2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model open_match --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 \
--lambda_oem 0.1 --lambda_socr 0.5'
screen -dmS om3 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model open_match --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 \
--lambda_oem 0.1 --lambda_socr 0.5'
screen -dmS om4 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model open_match --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 \
--lambda_oem 0.1 --lambda_socr 0.5'
screen -dmS om5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model open_match --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000 \
--lambda_oem 0.1 --lambda_socr 0.5'

screen -dmS osce1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce3 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .5 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .75 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'




# CIFAR100 EXP

screen -dmS vce bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/ce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model cr_ent --ood_ratio 1.0 --label_classes 55 --n_unlabels 20000 --strong_aug'
screen -dmS orce bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 1.0 --label_classes 55 --n_unlabels 20000 --strong_aug'
screen -dmS osce bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 55 --n_unlabels 20000 --strong_aug'


screen -dmS orce1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.0 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.25 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.5 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS orce4 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/osce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model oracle_cr_ent --ood_ratio 0.75 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.0 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.25 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .5 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'
screen -dmS osce4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .75 --label_classes 50 --n_unlabels 20000 --epochs 100 --strong_aug'


screen -dmS osce1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .5 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
screen -dmS osce4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/orce --arch wideresnet \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 2 --model os_cr_ent --ood_ratio .75 --label_classes 6 --n_unlabels 20000 --epochs 100 --strong_aug --exclude_ood'
