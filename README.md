# invariancetest

每次上传到drive的部分:
1. model_label.txt
2. robustacc.txt
3. plot里的mid文件夹

本次任务：
1. 更新train.py
2. 训练、上传的mid：32，51-65

具体细则如下：
32 retrain: (batch size改一下改到128或者64)
python train.py --mid=32 --epoch=50 --max_angle=15 --batch_size=128 --lr=0.0001 --anomaly=4 --pretrain=True

51-65: data leakage组，见run_this.txt文件




overleaf:
https://www.overleaf.com/project/60357cbfb6dfc5402b8ad95e
