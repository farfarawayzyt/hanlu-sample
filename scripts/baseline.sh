screen -dmS vce1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/ce --arch wideresnet --epochs 100 \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --mu 1 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --strong_aug --no_test_ood --save_model'
screen -dmS vce2 bash -c 'CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar10 --num-labeled 0 --out checkpoint/ce --arch wideresnet --epochs 100 \
--batch-size 128 --lr 0.03 --seed 0 --mu 1 --model cr_ent --ood_ratio 1.0 --label_classes 6 --n_unlabels 20000 --strong_aug --no_test_ood --save_model'

screen -dmS fix1 bash -c 'CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar10 --num-labeled 50 --out checkpoint/fix --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --threshold 0.95 --seed 0 --mu 2 --model fix_match --label_classes 6 --n_unlabels 20000 --no_test_ood --save_model'