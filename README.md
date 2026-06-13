基于深度学习和传统机器学习的新闻推荐系统，包含召回和排序两个阶段。

## 项目结构

数据需要自行到比赛页面下载，放入Initial_data目录下。

```
.
├── Initial_data/           # 原始数据目录
├── Recall/                 # 召回模块
│   ├── DSSM_recall.py      # DSSM深度语义召回
│   ├── Recall_itemcf.py    # ItemCF协同过滤
│   ├── Recall_Methods.py   # 评估工具
│   ├── recall_data_gen.py  # 召回数据生成
│   └── merge.py            # 多路召回合并
├── Rank/                   # 排序模块
│   ├── DCN_rank.py         # DCN深度交叉网络
│   ├── din_rank.py         # DIN深度兴趣网络
│   └── Feat_Eng.py         # 特征工程
├── util/                   # 工具模块
│   └── utils.py
├── requirements.txt        # 依赖配置
└── run_all.bat            # 批量运行脚本
```

## 环境配置
```bash
# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 召回
python ./Recall/Recall_itemcf.py
python ./Recall/DSSM_recall.py
python ./Recall/Recall_merge.py

# 2. 特征工程
python ./Rank/Feat_Eng.py

# 3. 排序
python ./Rank/DCN_rank.py
python ./Rank/din_rank.py
```

## 参考说明

本项目在Datawhale提供的开源思路基础上改进。
