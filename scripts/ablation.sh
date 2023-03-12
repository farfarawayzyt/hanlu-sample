
# bc temperature
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 0.07 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 0.2 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 1.0 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'


# cosine temperature
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type cos --contrast_T 0.07 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type cos --contrast_T 0.2 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type cos --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type cos --contrast_T 1.0 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp --dummy_classes 4 --no_test_ood'


# bc ema_strong

screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 0.2 --epochs 512 --ema_strong \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp_ema_strong --dummy_classes 4 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type bc --contrast_T 0.5 --epochs 512 --ema_strong \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp_ema_strong --dummy_classes 4 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 --ema_strong \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des bc_contrastive_exp_ema_strong --dummy_classes 4 --no_test_ood'


# dummy class
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 2 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 4 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 8 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 16 --no_test_ood'

# temperature
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.05 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.1 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.2 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 2.0 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'

# batch size
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 1 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 2 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 8 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 16 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'

# threshold
screen -dmS con1 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.4 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.6 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.8 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'
screen -dmS con4 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 512 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 1.0 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des new_benchmark --dummy_classes 1 --no_test_ood'


# varying ood ratio
screen -dmS ratio1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 200 --ood_ratio 0.0 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ablation_ood --dummy_classes 4 --vary_ratio --no_test_ood'
screen -dmS ratio2 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 200 --ood_ratio 0.3 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ablation_ood --dummy_classes 4 --vary_ratio --no_test_ood'
screen -dmS ratio3 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 200 --ood_ratio 0.6 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ablation_ood --dummy_classes 4 --vary_ratio --no_test_ood'
screen -dmS ratio4 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 200 --ood_ratio 0.9 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 4 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ablation_ood --dummy_classes 4 --vary_ratio --no_test_ood'
screen -dmS ratio3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 4 --out checkpoint/fix --arch wideresnet --sim_type ce --contrast_T 0.5 --epochs 200 --ood_ratio 0.9 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.9 --seed 0 --mu 7 --model fix_contrast --label_classes 6 --n_unlabels 20000 --des ablation_ood --dummy_classes 4 --vary_ratio --no_test_ood'


screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.0 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.3 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.6 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 100 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.9 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'


screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.0 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0 --seed 0 --mu 2 --model open_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix2 bash -c 'CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.3 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0 --seed 0 --mu 2 --model open_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix3 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.6 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0 --seed 0 --mu 2 --model open_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'
screen -dmS fix4 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 200 --ood_ratio 0.9 \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0 --seed 0 --mu 2 --model open_match --label_classes 6 --n_unlabels 20000 --no_test_ood --vary_ratio'