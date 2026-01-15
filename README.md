# badminton pred

## How to run
create the following folders
```
logs/*
models/*
results/*
visualization/*
```
download dataset from lk:
```
20250809_Seq_data/*
```
finally, run model:
```
python main.py
```

The experimental results are recorded in [this FeiShu link](https://eve0wcbe0c.feishu.cn/wiki/EvigwZ0OriBi6IkB2U1cNmSRnid?from=from_copylink)

NOTE:
If you use the server (10.32.80.5:22), just git clone this repo:
```
git clone /home/zhaoxuhao/git/repo/badminton.git
```

### ================== WangYiHeng分支的更新 ===================

首先运行完成模型的训练
```angular2html
python main.py
```
程序会将训练完成的模型在测试样本上进行测试，并将全部测试样本及指标存入csv文件中

随后可以运行下面的程序进行进一步的可视化（包括不确定性估计的可视化）
```angular2html
python visual_csv.py
```

具体的不确定性估计方法参见 [Feishu link](https://icnw67607rq8.feishu.cn/wiki/VSKRwBaWwi1c92ksYA3cjvrJnde)
的 11.14 日期相关内容


### ================== Mushroom-cat分支的更新 ===================

已加入滑动窗口训练，min_offset_len、max_offset_len分别对应训练时随机的最小offset、最大offset。注意，由于最后一帧是击球后5帧，当你需要训练击球前模型的时候应该改成5、25，当你需要训练击球后模型的时候应该改成0、4。

temp_test_offset是用于固定测试时使用的offset的，用于画特定offset时的模型效果图。注意这个参数只固定测试时的offset，训练时的offset一直是随机的。如果这个参数<0，则会在测试时也使用随机offset（随机范围和训练的一样）。

main_tryoffset.py是用来画不确定性-实际误差的散点图的。

1.sh是用来批量实验的。