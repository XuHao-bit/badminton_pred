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