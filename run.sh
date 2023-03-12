# fixmatch
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 2 \
#   --batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --eval-step 512
CUDA_VISIBLE_DEVICES=0 python main.py --dataset imagenet --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet --epochs 1 \
  --batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --eval-step 512

# simclr

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet --epochs 2 \
#   --batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model simclr --label_classes 6 --n_unlabels 20000 --no_test_ood --eval-step 512
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset imagenet --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet --epochs 2 \
#   --batch-size 64 --lr 0.03 --expand-labels --seed 0 --mu 2 --model simclr --label_classes 6 --n_unlabels 20000 --no_test_ood --eval-step 512
