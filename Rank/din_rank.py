"""
DIN (Deep Interest Network) 排序模型

功能：使用DIN深度学习模型对召回候选进行精排
核心特性：
1. 引入注意力机制捕捉用户历史行为与候选商品的相关性
2. 支持变长序列输入（用户历史点击序列）
3. 使用5折交叉验证进行训练
4. 对Dense特征进行分桶离散化处理
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models import DIN
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Recall.Recall_Methods import evaluate
from util.utils import Logger

warnings.filterwarnings('ignore')


# ==================== 路径配置 ====================
# 相对于 Code/Rank 目录的路径
FEAT_ENG_PATH = Path('../Results/Feat_ENG')
DATA_PATH = Path('../Initial_data')
SUBMIT_PATH = Path('../Results/submit')
LOG_PATH = Path('../Results/log')


# ==================== 参数配置 ====================
RECALL_NUM = 50         # 召回阶段保留的候选数量
MAX_LEN = 5             # 用户历史行为序列的最大长度
EMB_DIM = 32            # Embedding维度
N_BINS = 100            # Dense特征分桶数量
K_FOLD = 5              # 交叉验证折数
BATCH_SIZE = 256        # 训练批次大小
EPOCHS = 10             # 训练轮数
PATIENCE = 5            # 早停耐心值

# ==================== DEBUG模式配置 ====================
# DEBUG_MODE = True 时，只使用少量数据快速验证代码正确性
DEBUG_MODE = False
DEBUG_MAX_USERS = 1000      # DEBUG模式下最大用户数量
DEBUG_EPOCHS = 1            # DEBUG模式下训练轮数


def setup_logger():
    """
    初始化日志记录器

    Returns:
        logger: 配置好的日志记录器
    """
    curtime = datetime.now().strftime('%Y%m%d')
    curfile = Path(__file__).stem
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_file = str(LOG_PATH / f'test_{curfile}_{curtime}.txt')
    return Logger(log_file).logger


def load_feature_data(recall_num):
    """
    加载特征工程生成的特征数据

    Parameters:
    -----------
    recall_num : int
        召回候选数量（用于定位特征文件）

    Returns:
    --------
    tuple: (df_feature, feat_cols)
        - df_feature: 完整的特征DataFrame
        - feat_cols: 用于模型的特征列名列表
    """
    df_feature = pd.read_csv(FEAT_ENG_PATH / f'df_feature_{recall_num}.csv')

    # 检查并处理NaN和Inf值，防止训练时出现NaN
    numeric_cols = df_feature.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 替换Inf为NaN，然后填充NaN
        df_feature[col] = df_feature[col].replace([np.inf, -np.inf], np.nan)
        if df_feature[col].isna().any():
            median_val = df_feature[col].median()
            df_feature[col] = df_feature[col].fillna(median_val)

    # 排除非特征列（标签、时间、相似度分数等）
    exclude_cols = ['label', 'created_at_datetime', 'click_datetime', 'sim_score']
    feat_cols = [col for col in df_feature.columns if col not in exclude_cols]

    return df_feature, feat_cols


def split_train_test(df_feature, debug_mode=False, debug_max_users=None):
    """
    将数据划分为训练集和测试集

    处理逻辑：
    1. 训练集：user_id < 200000，且只保留有正样本的用户
    2. 测试集：user_id >= 200000
    3. DEBUG模式：限制训练集用户数量以加速测试

    Parameters:
    -----------
    df_feature : pd.DataFrame
        完整的特征数据
    debug_mode : bool
        是否开启DEBUG模式
    debug_max_users : int
        DEBUG模式下最大用户数量

    Returns:
    --------
    tuple: (trn_feats_df, tst_feats_df, total_users)
        - trn_feats_df: 训练集
        - tst_feats_df: 测试集
        - total_users: 训练集用户数量（用于评估）
    """
    # 划分训练集和测试集
    trn_feats_df = df_feature[df_feature['user_id'] < 200000].copy()
    tst_feats_df = df_feature[df_feature['user_id'] >= 200000].copy()

    # 训练集：只保留有正样本的用户（用于训练排序模型）
    valid_user_ids = trn_feats_df[trn_feats_df['label'] == 1]['user_id'].unique()
    trn_feats_df = trn_feats_df[trn_feats_df['user_id'].isin(valid_user_ids)].copy()

    # DEBUG模式：限制用户数量
    if debug_mode and debug_max_users:
        all_users = trn_feats_df['user_id'].unique()
        selected_users = all_users[:min(debug_max_users, len(all_users))]
        trn_feats_df = trn_feats_df[trn_feats_df['user_id'].isin(selected_users)].copy()
        # 测试集也限制
        tst_users = tst_feats_df['user_id'].unique()[:min(debug_max_users // 2, len(tst_feats_df['user_id'].unique()))]
        tst_feats_df = tst_feats_df[tst_feats_df['user_id'].isin(tst_users)].copy()
        print(f"[DEBUG模式] 训练集用户数限制为: {len(selected_users)}")
        print(f"[DEBUG模式] 测试集用户数限制为: {len(tst_users)}")

    # 按用户和相似度分数排序
    trn_feats_df = trn_feats_df.sort_values(
        by=['user_id', 'sim_score'],
        ascending=[True, False]
    ).reset_index(drop=True)

    tst_feats_df = tst_feats_df.sort_values(
        by=['user_id', 'sim_score'],
        ascending=[True, False]
    ).reset_index(drop=True)

    total_users = trn_feats_df['user_id'].nunique()

    return trn_feats_df, tst_feats_df, total_users


def evaluate_before_training(trn_feats_df, total_users, log):
    """
    训练前评估：使用召回阶段的相似度分数作为基准

    Parameters:
    -----------
    trn_feats_df : pd.DataFrame
        训练集特征
    total_users : int
        总用户数
    log : logger
        日志记录器
    """
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, \
        hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            trn_feats_df, total_users, 'article_id'
        )

    log.info('训练前评估（召回阶段分数）：\n'
             f'HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}\n'
             f'HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}\n'
             f'HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}\n'
             f'HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}\n'
             f'HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}')


def build_click_history_features(trn_feats_df, tst_feats_df):
    """
    构建用户历史点击行为特征

    为DIN模型准备用户历史行为序列，用于注意力机制计算
    历史序列按时间倒序排列（最近点击的文章在前）

    Parameters:
    -----------
    trn_feats_df : pd.DataFrame
        训练集特征
    tst_feats_df : pd.DataFrame
        测试集特征

    Returns:
    --------
    tuple: (trn_feats_df_din, tst_feats_df_din)
        合并了历史行为序列的特征DataFrame
    """
    # 加载原始点击数据
    trn_data = pd.read_csv(DATA_PATH / 'train_click_log.csv')
    tst_data = pd.read_csv(DATA_PATH / 'testA_click_log.csv')

    # 按用户和时间排序
    trn_data = trn_data.sort_values(
        by=['user_id', 'click_timestamp'],
        ascending=[True, True]
    )

    # 训练集：排除每个用户的最后一次点击（作为验证标签）
    trn_hist_df = trn_data.groupby('user_id').apply(
        lambda x: x.iloc[:-1]
    ).reset_index(drop=True)

    # 按时间倒序排列（最近点击在前）
    trn_hist_df = trn_hist_df.sort_values(
        by=['user_id', 'click_timestamp'],
        ascending=[True, False]
    ).reset_index(drop=True)

    # 测试集：保留全部点击记录
    tst_hist_df = tst_data.sort_values(
        by=['user_id', 'click_timestamp'],
        ascending=[True, False]
    ).reset_index(drop=True)

    # 构建训练集历史行为序列
    trn_hist_click = trn_hist_df.groupby('user_id')['click_article_id'].agg(list).reset_index()
    trn_his_behavior_df = pd.DataFrame({
        'user_id': trn_hist_click['user_id'],
        'hist_article_id': trn_hist_click['click_article_id']
    })

    # 构建测试集历史行为序列
    tst_hist_click = tst_hist_df.groupby('user_id')['click_article_id'].agg(list).reset_index()
    tst_his_behavior_df = pd.DataFrame({
        'user_id': tst_hist_click['user_id'],
        'hist_article_id': tst_hist_click['click_article_id']
    })

    # 合并历史行为特征到主表
    trn_feats_df_din = trn_feats_df.merge(trn_his_behavior_df, on='user_id', how='left')
    tst_feats_df_din = tst_feats_df.merge(tst_his_behavior_df, on='user_id', how='left')

    return trn_feats_df_din, tst_feats_df_din


def discretize_dense_features(trn_df, tst_df, dense_fea, n_bins=100):
    """
    将Dense特征进行分桶离散化处理

    DIN模型通常将所有特征视为Sparse特征处理。
    这里使用分位数分桶（quantile-based binning）将连续值转为离散类别。

    Parameters:
    -----------
    trn_df : pd.DataFrame
        训练集DataFrame（会被修改）
    tst_df : pd.DataFrame
        测试集DataFrame（会被修改）
    dense_fea : list
        需要离散化的Dense特征列名列表
    n_bins : int
        分桶数量

    Returns:
    --------
    dict: bin_edges - 每个特征的分桶边界，可用于后续新数据转换
    """
    bin_edges = {}

    for feat in dense_fea:
        # 基于训练集的非NaN值计算分位数边界
        _, bin_edges[feat] = pd.qcut(
            trn_df[feat].dropna(),
            q=n_bins,
            retbins=True,
            duplicates="drop"  # 处理重复分位数
        )

        # 对训练集应用分桶
        trn_df[feat] = pd.cut(
            trn_df[feat],
            bins=bin_edges[feat],
            labels=False,
            include_lowest=True
        ).fillna(n_bins).astype(int)  # NaN值分配到额外的一个桶

        # 对测试集应用相同的分桶边界
        tst_df[feat] = pd.cut(
            tst_df[feat],
            bins=bin_edges[feat],
            labels=False,
            include_lowest=True
        ).fillna(n_bins).astype(int)

    return bin_edges


def validate_data(trn_df, tst_df, dense_fea, sparse_fea, behavior_fea):
    """
    验证输入数据是否包含NaN或Inf值

    Parameters:
    -----------
    trn_df, tst_df : pd.DataFrame
        训练集和测试集
    dense_fea, sparse_fea, behavior_fea : list
        各类特征列名
    """
    # 检查训练集
    assert not trn_df[dense_fea + sparse_fea].isnull().any().any(), "训练集包含NaN值"
    assert not np.isinf(trn_df[dense_fea + sparse_fea + behavior_fea].values).any(), "训练集包含Inf值"

    # 检查测试集
    assert not tst_df[dense_fea + sparse_fea].isnull().any().any(), "测试集包含NaN值"
    assert not np.isinf(tst_df[dense_fea + sparse_fea + behavior_fea].values).any(), "测试集包含Inf值"


# ++ 新增函数：基于全量数据统一构建特征列（只需调用一次）
def build_feature_columns(full_df, dense_fea, sparse_fea, his_behavior_fea,
                          emb_dim=32, max_len=100):
    """
    构建DIN模型的特征列配置

    DIN模型需要三类特征：
    1. SparseFeat: 离散型特征（如user_id, article_id）
    2. DenseFeat: 数值型特征（分桶后转为sparse处理）
    3. VarLenSparseFeat: 变长序列特征（用户历史行为序列）

    vocab_size 必须基于全量数据（训练集 + 测试集）统一计算，
    确保所有数据子集（各折训练集、验证集、测试集）使用相同的
    embedding 维度，避免 embedding 越界或各子集维度不一致的问题。

    Parameters:
    -----------
    full_df : pd.DataFrame
        全量数据（训练集 + 测试集合并后）
    dense_fea : list
        数值型特征列
    sparse_fea : list
        离散型特征列
    his_behavior_fea : list
        历史行为特征列（变长序列）
    emb_dim : int
        Embedding维度
    max_len : int
        序列最大长度（用于padding）

    Returns:
    --------
    list: dnn_feature_columns - 特征列配置列表
    """
    # 用最大索引值计算词表大小，避免ID非连续时出现embedding越界
    sparse_vocab_sizes = {}
    for feat in sparse_fea:
        feat_max = pd.to_numeric(full_df[feat], errors='coerce').max()
        feat_max = 0 if pd.isna(feat_max) else int(feat_max)
        sparse_vocab_sizes[feat] = max(feat_max + 1, 2)

    # article_id 和 hist_article_id 共享 embedding，词表需覆盖两者最大 ID
    article_max = pd.to_numeric(full_df['article_id'], errors='coerce').max()
    article_max = 0 if pd.isna(article_max) else int(article_max)
    hist_max = pd.to_numeric(full_df['hist_article_id'].explode(), errors='coerce').max()
    hist_max = 0 if pd.isna(hist_max) else int(hist_max)
    article_vocab_size = max(article_max, hist_max) + 1

    # Sparse特征：为每个离散特征创建Embedding配置
    sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=sparse_vocab_sizes[feat], embedding_dim=emb_dim)
        for feat in sparse_fea
    ]

    # Dense特征（此时已被离散化）
    dense_feature_columns = [
        DenseFeat(feat, 1) for feat in dense_fea
    ]

    # 变长序列特征：用户历史行为序列
    # 注意：所有历史序列共享article_id的Embedding空间
    var_feature_columns = [
        VarLenSparseFeat(
            SparseFeat(
                feat,
                vocabulary_size=article_vocab_size,
                embedding_dim=emb_dim,
                embedding_name='article_id'  # 与候选article_id共享Embedding
            ),
            maxlen=max_len
        )
        for feat in his_behavior_fea
    ]

    # 合并所有特征列
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns

    return dnn_feature_columns


# ++ 新增函数：基于数据子集构建模型输入字典
def build_model_input(df, dnn_feature_columns, his_behavior_fea, max_len=100):
    """
    基于给定数据子集构建模型输入字典

    feature_columns 已在外部由 build_feature_columns 统一生成并固定，
    此函数只负责从 df 中取值并对序列特征做 padding。

    Parameters:
    -----------
    df : pd.DataFrame
        当前数据子集（训练集/验证集/测试集）
    dnn_feature_columns : list
        特征列配置（由 build_feature_columns 统一生成）
    his_behavior_fea : list
        历史行为特征列（变长序列）
    max_len : int
        序列最大长度

    Returns:
    --------
    dict: 模型输入字典 x
    """
    # 构建模型输入字典
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 历史行为序列需要padding到固定长度
            his_list = df[name].tolist()
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')
        else:
            x[name] = df[name].values

    return x


def get_kfold_users(trn_df, n=5):
    """
    将用户划分为n折，用于交叉验证

    Parameters:
    -----------
    trn_df : pd.DataFrame
        训练集
    n : int
        折数

    Returns:
    --------
    list: 每折的用户ID数组列表
    """
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


def train_din_model(trn_df, tst_df, sparse_fea, dense_fea, hist_behavior_fea,
                    behavior_fea, max_len, log, debug_mode=False, debug_epochs=None):
    """
    使用5折交叉验证训练DIN模型

    Parameters:
    -----------
    trn_df : pd.DataFrame
        训练集
    tst_df : pd.DataFrame
        测试集（用于预测）
    sparse_fea, dense_fea, hist_behavior_fea, behavior_fea : list
        各类特征列
    max_len : int
        序列最大长度
    log : logger
        日志记录器
    debug_mode : bool
        DEBUG模式标志
    debug_epochs : int
        DEBUG模式下的训练轮数

    Returns:
    --------
    tuple: (df_oof, prediction)
        - df_oof: 训练集OOF预测结果
        - prediction: 测试集预测结果
    """
    # ++ 新增：合并全量数据，基于全量统一计算一次 feature_columns
    full_df = pd.concat([trn_df, tst_df], axis=0, ignore_index=True)
    dnn_feature_columns = build_feature_columns(
        full_df, dense_fea, sparse_fea, hist_behavior_fea, max_len=max_len
    )

    # ++ 修改：测试集输入只调用 build_model_input，feature_columns 复用上面统一生成的
    # -- 删除：x_tst, dnn_feature_columns = get_din_feature_columns(
    # --           tst_df, dense_fea, sparse_fea, hist_behavior_fea, max_len=max_len)
    x_tst = build_model_input(tst_df, dnn_feature_columns, hist_behavior_fea, max_len=max_len)

    # 初始化预测结果容器
    prediction = tst_df[['user_id', 'article_id']].copy()
    prediction['pred'] = 0
    oof_list = []

    # 获取交叉验证的用户划分
    user_set = get_kfold_users(trn_df, n=K_FOLD)

    # DEBUG模式下减少折数
    if debug_mode:
        user_set = user_set[:2]  # 只使用2折
        log.info(f"[DEBUG模式] 交叉验证折数限制为: {len(user_set)}")

    # 每折训练和预测
    for n_fold, valid_users in enumerate(user_set):
        log.info(f'\n===== Fold {n_fold + 1}/{len(user_set)} =====')

        # 划分训练和验证集（按用户划分，避免数据泄露）
        train_idx = trn_df[~trn_df['user_id'].isin(valid_users)]
        valid_idx = trn_df[trn_df['user_id'].isin(valid_users)]

        # ++ 修改：只调用 build_model_input，复用统一生成的 feature_columns
        # -- 删除：x_trn, _ = get_din_feature_columns(
        # --           train_idx, dense_fea, sparse_fea, hist_behavior_fea, max_len=max_len)
        x_trn = build_model_input(train_idx, dnn_feature_columns, hist_behavior_fea, max_len=max_len)
        y_trn = train_idx['label'].values

        # -- 删除：x_val, _ = get_din_feature_columns(
        # --           valid_idx, dense_fea, sparse_fea, hist_behavior_fea, max_len=max_len)
        x_val = build_model_input(valid_idx, dnn_feature_columns, hist_behavior_fea, max_len=max_len)
        y_val = valid_idx['label'].values

        # 构建DIN模型
        model = DIN(dnn_feature_columns, behavior_fea)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_crossentropy', tf.keras.metrics.AUC()]
        )

        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        # NaN检测回调
        terminate_on_nan = TerminateOnNaN()

        # DEBUG模式下使用更少的epoch
        epochs = debug_epochs if debug_mode else EPOCHS

        # 训练
        model.fit(
            x_trn, y_trn,
            verbose=1,
            epochs=epochs,
            validation_data=(x_val, y_val),
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, terminate_on_nan]
        )

        # 验证集预测（用于OOF评估）
        pred_val = model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)
        df_oof = valid_idx[['user_id', 'article_id', 'label']].copy()
        df_oof['pred'] = pred_val
        oof_list.append(df_oof)

        # 测试集预测（取平均）
        pred_test = model.predict(x_tst, verbose=1, batch_size=BATCH_SIZE)[:, 0]
        prediction['pred'] += pred_test / K_FOLD

    # 合并所有折的OOF结果
    df_oof = pd.concat(oof_list, ignore_index=True)

    return df_oof, prediction


def evaluate_din_results(df_oof, log):
    """
    评估DIN模型的排序效果

    Parameters:
    -----------
    df_oof : pd.DataFrame
        OOF预测结果
    log : logger
        日志记录器
    """
    # 按预测分数排序
    df_oof = df_oof.sort_values(
        by=['user_id', 'pred'],
        ascending=[True, False]
    ).reset_index(drop=True)

    log.info(f'OOF预测结果预览:\n{df_oof.head()}')

    # 计算评估指标
    total_users = df_oof['user_id'].nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, \
        hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_oof, total_users, 'article_id'
        )

    log.info('DIN模型评估结果：\n'
             f'HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}\n'
             f'HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}\n'
             f'HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}\n'
             f'HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}\n'
             f'HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}')


def generate_submission(prediction, submit_path, recall_num, curtime, log):
    """
    生成提交文件

    Parameters:
    -----------
    prediction : pd.DataFrame
        测试集预测结果
    submit_path : Path
        提交文件保存路径
    recall_num : int
        召回数量（用于文件名）
    curtime : str
        当前时间（用于文件名）
    log : logger
        日志记录器
    """
    # 去重并排序
    prediction = prediction.drop_duplicates()
    prediction = prediction.sort_values(
        by=['user_id', 'pred'],
        ascending=[True, False]
    )

    # 取每个用户的top5
    df_top5 = prediction.groupby('user_id').head(5).reset_index(drop=True)

    # 添加排名
    df_top5['rank'] = df_top5.groupby('user_id')['pred'].rank(
        method='first',
        ascending=False
    )

    # 将长格式转为宽格式（每个用户一行）
    df_pivot = df_top5.pivot(index='user_id', columns='rank', values='article_id')
    df_pivot.columns = df_pivot.columns.astype(int)
    df_pivot = df_pivot.rename(columns={
        1: 'article_1',
        2: 'article_2',
        3: 'article_3',
        4: 'article_4',
        5: 'article_5'
    })
    df_pivot = df_pivot.reset_index()

    log.info(f'提交文件预览:\n{df_pivot}')

    # 保存提交文件
    submit_path.mkdir(parents=True, exist_ok=True)
    output_file = submit_path / f'din_{recall_num}_{curtime}.csv'
    df_pivot.to_csv(output_file, index=False, encoding='utf-8-sig')
    log.info(f'提交文件已保存: {output_file}')


def main():
    """主函数"""
    # ==================== 1. 初始化 ====================
    log = setup_logger()
    log.info("=" * 60)
    log.info("DIN排序模型启动")
    if DEBUG_MODE:
        log.info("[DEBUG模式开启] 使用少量数据快速验证代码")
    log.info("=" * 60)

    # ==================== 2. 加载特征数据 ====================
    log.info("【步骤1】加载特征工程数据...")
    df_feature, feat_cols = load_feature_data(RECALL_NUM)
    log.info(f"特征列: {feat_cols}")
    log.info(f"特征数据形状: {df_feature.shape}")

    # ==================== 3. 划分训练集/测试集 ====================
    log.info("【步骤2】划分训练集和测试集...")
    trn_feats_df, tst_feats_df, total_users = split_train_test(
        df_feature,
        debug_mode=DEBUG_MODE,
        debug_max_users=DEBUG_MAX_USERS if DEBUG_MODE else None
    )
    log.info(f"训练集用户数: {total_users}")
    log.info(f"训练集样本数: {len(trn_feats_df)}")

    # ==================== 4. 训练前评估（基准） ====================
    log.info("【步骤3】训练前评估（召回分数作为基准）...")
    evaluate_before_training(trn_feats_df, total_users, log)

    # ==================== 5. 构建历史行为序列 ====================
    log.info("【步骤4】构建用户历史点击行为序列...")
    trn_feats_df_din, tst_feats_df_din = build_click_history_features(
        trn_feats_df, tst_feats_df
    )

    # ==================== 6. 特征类型定义 ====================
    # 基础Sparse特征（始终为离散型）
    base_sparse_fea = ['user_id', 'article_id', 'category_id']
    # 候选文章行为特征（用于DIN的注意力机制）
    behavior_fea = ['article_id']
    # 历史行为序列特征
    hist_behavior_fea = ['hist_article_id']
    # Dense特征（需要分桶离散化）
    dense_fea = [col for col in feat_cols if col not in base_sparse_fea]

    # 确保dense特征为数值类型
    for col in dense_fea:
        trn_feats_df_din[col] = trn_feats_df_din[col].astype('float64')
        tst_feats_df_din[col] = tst_feats_df_din[col].astype('float64')

    # ==================== 7. Dense特征分桶离散化 ====================
    log.info("【步骤5】Dense特征分桶离散化...")
    discretize_dense_features(
        trn_feats_df_din, tst_feats_df_din, dense_fea, n_bins=N_BINS
    )

    # 填充category_id的缺失值
    tst_feats_df_din['category_id'] = tst_feats_df_din['category_id'].fillna(-1)
    trn_feats_df_din['category_id'] = trn_feats_df_din['category_id'].fillna(-1)

    # ==================== 8. 数据验证 ====================
    log.info("【步骤6】验证数据完整性...")
    validate_data(
        trn_feats_df_din, tst_feats_df_din,
        dense_fea, base_sparse_fea, behavior_fea
    )

    # 分桶后，所有特征都作为sparse处理
    sparse_fea = base_sparse_fea + dense_fea
    dense_fea = []

    # ==================== 9. DIN模型训练 ====================
    log.info("【步骤7】训练DIN模型（5折交叉验证）...")
    df_oof, prediction = train_din_model(
        trn_feats_df_din, tst_feats_df_din,
        sparse_fea, dense_fea, hist_behavior_fea,
        behavior_fea, MAX_LEN, log,
        debug_mode=DEBUG_MODE,
        debug_epochs=DEBUG_EPOCHS if DEBUG_MODE else None
    )

    # ==================== 10. 模型评估 ====================
    log.info("【步骤8】评估DIN模型效果...")
    evaluate_din_results(df_oof, log)

    # ==================== 11. 生成提交文件 ====================
    log.info("【步骤9】生成提交文件...")
    curtime = datetime.now().strftime('%Y%m%d')
    if DEBUG_MODE:
        curtime = f"{curtime}_debug"
    generate_submission(prediction, SUBMIT_PATH, RECALL_NUM, curtime, log)

    log.info("=" * 60)
    log.info("DIN排序模型运行完成")
    log.info("=" * 60)


if __name__ == "__main__":
    main()