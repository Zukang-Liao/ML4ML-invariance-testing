# invariancetest

下一次上传到drive的部分:
1. mid: 32, 46-50(上一次plots里面的部分没有上传成功)
2. 内容: model_label.txt, robustacc.txt 以及plot里面对应的mid部分

32 retrain:
python train.py --mid=32 --epoch=50 --max_angle=15 --batch_size=128 --lr=0.0001 --anomaly=4 --pretrain=True
