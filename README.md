# invariancetest

下一次上传到drive的部分:
1. mid: 32, 51-65
2. 内容: model_label.txt, robustacc.txt 以及plot里面对应的mid部分

32 retrain: (batch size改一下改到128或者64)
python train.py --mid=32 --epoch=50 --max_angle=15 --batch_size=128 --lr=0.0001 --anomaly=4 --pretrain=True

data leakage组
python train.py --mid=51 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.05

python train.py --mid=52 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1

python train.py --mid=53 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.2

python train.py --mid=54 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.3

python train.py --mid=55 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.5

python train.py --mid=56 --epoch=50 --max_angle=5 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1

python train.py --mid=57 --epoch=50 --max_angle=20 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1

python train.py --mid=58 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --r=0.1

python train.py --mid=59 --epoch=50 --max_angle=5 --batch_size=1024 --lr=0.0001 --anomaly=8 --r=0.1

python train.py --mid=60 --epoch=50 --max_angle=20 --batch_size=1024 --lr=0.0001 --anomaly=8 --r=0.1

python train.py --mid=61 --epoch=50 --max_angle=5 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1 --adv=True --epsilon=0.015625

python train.py --mid=62 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1 --adv=True --epsilon=0.015625

python train.py --mid=63 --epoch=50 --max_angle=20 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.1 --adv=True --epsilon=0.015625

python train.py --mid=64 --epoch=50 --max_angle=5 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.5 --adv=True --epsilon=0.015625

python train.py --mid=65 --epoch=50 --max_angle=15 --batch_size=1024 --lr=0.0001 --anomaly=8 --pretrain=True --r=0.5 --adv=True --epsilon=0.015625



overleaf:
https://www.overleaf.com/project/60357cbfb6dfc5402b8ad95e
