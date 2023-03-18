python dali_main.py --dataset imagenet --num-labeled 50 --out checkpoint/fix --arch resnet_imagenet \
    --epochs 1 --batch-size 32 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 \
    --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --eval-step 512
