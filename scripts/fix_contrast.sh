screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --des bc_contrast'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --des bc_contrast'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --des bc_contrast'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 4 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --des bc_contrast_dummy4'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 4 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --des bc_contrast_dummy4'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 4 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --des bc_contrast_dummy4'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000 --sim_type cos'
screen -dmS fix6 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000 --proj_head'
screen -dmS fix7 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000 --memory_bank'

# CIFAR100

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.0 --label_classes 6 --n_unlabels 20000 --proj_head'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.25 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.5 --label_classes 6 --n_unlabels 20000 --proj_head'
#screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.75 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --proj_head'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000 --memory_bank'
#screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
#--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.3 --label_classes 6 --n_unlabels 20000 --memory_bank'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet --dummy_classes 1 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --ood_ratio 0.6 --label_classes 6 --n_unlabels 20000 --memory_bank'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 400 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 60 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --num-labeled 100 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 60 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --num-labeled 400 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 60 --n_unlabels 20000 --des new_benchmark'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --no_sg --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --soft_mask --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --contrast_neg --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'

screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_neg --contrast_self --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_self --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 1.0 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'

# ablation
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --soft_mask --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_neg --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --soft_mask --contrast_neg --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --soft_mask --contrast_neg --epochs 150 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark'
# OOD exp

screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --open_split'
screen -dmS fix5 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --sim_type ce --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ood_detect --open_split --contrast_T 0.2'

# imagenet30
python main.py --dataset imagenet --out $1 --arch resnet_imagenet --lambda_oem 0.1 --lambda_socr 0.5 \
  --batch-size 64 --lr 0.03 --expand-labels --seed 0 --opt_level O2 --amp --mu 2 --epochs 100

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_contrast --des ood_detect --contrast_T 0.2'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --sim_type bc --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_contrast --des ood_detect --contrast_T 0.2'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_contrast --des ood_detect --contrast_T 0.5'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --sim_type bc --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_contrast --des ood_detect --contrast_T 0.5'


screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset imagenet  --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet \
--dummy_classes 1 --batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_match_uniform'



