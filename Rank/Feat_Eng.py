"""
特征工程模块 (Feature Engineering)

功能：为排序模型构建训练特征
处理流程：
1. 加载召回阶段的候选结果（训练集和测试集）
2. 构建用户历史行为特征（时间差、字数统计等）
3. 构建文章特征（创建时间、字数等）
4. 构建交叉特征（候选文章与用户历史行为的差异）
5. 保存特征文件供排序模型使用
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.utils import Logger

# 配置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


# ==================== 路径配置 ====================
# 相对于 Code/Rank 目录的路径
RECALL_PATH = Path('../Results/Recall_dict')
DATA_PATH = Path('../Initial_data')
SAVE_PATH = Path('../Results/Feat_ENG')
LOG_PATH = Path('../Results/log')


# ==================== 参数配置 ====================
SEED = 2021
RECALL_NUM = 50  # 召回阶段保留的候选数量


def setup_logger():
    """初始化日志记录器"""
    now = datetime.now()
    formatted_date = now.strftime("%Y%m%d")
    current_file_name = Path(__file__).stem
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_file = str(LOG_PATH / f'test_{current_file_name}_{formatted_date}.txt')
    return Logger(log_file).logger


def consine_distance(vector1, vector2):
    """
    计算余弦相似度

    余弦相似度 = (向量点积) / (向量A模长 * 向量B模长)
    值域：[-1, 1]，1表示完全相似，-1表示完全不相似

    Parameters
    ----------
    vector1, vector2 : np.ndarray
        输入向量

    Returns
    -------
    float
        余弦相似度，输入类型错误时返回-1
    """
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return distance


def load_recall_candidates(recall_path, recall_num):
    """
    加载召回阶段的候选结果

    Parameters
    ----------
    recall_path : Path
        召回结果保存路径
    recall_num : int
        截断数量（取前recall_num个候选）

    Returns
    -------
    tuple: (df_train, df_tst, df_feature)
        - df_train: 训练集候选 (user_id < 200000)
        - df_tst: 测试集候选
        - df_feature: 合并后的特征DataFrame
    """
    # 加载训练集和测试集的召回结果
    df_train = pd.read_pickle(recall_path / f'train_final_recall_{recall_num}.pkl')
    df_tst = pd.read_pickle(recall_path / f'tst_final_recall_{recall_num}.pkl')

    # 训练集只保留user_id < 200000的用户（留出200000+作为验证集）
    df_train = df_train[df_train['user_id'] < 200000].reset_index(drop=True)

    # 合并训练集和测试集一起进行特征工程
    df_feature = pd.concat([df_train, df_tst], axis=0)

    return df_train, df_tst, df_feature


def load_article_features(data_path):
    """
    加载文章基础特征

    Parameters
    ----------
    data_path : Path
        数据路径

    Returns
    -------
    pd.DataFrame
        文章特征表，包含created_at_ts、words_count等
    """
    df_article = pd.read_csv(data_path / 'articles.csv')

    # 时间戳从毫秒转换为秒（原始数据是毫秒级时间戳）
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')

    return df_article


def build_click_history(data_path, df_article):
    """
    构建用户点击历史记录

    处理逻辑：
    1. 训练集：排除每个用户的最后一次点击（作为验证标签）
    2. 测试集：保留全部点击记录
    3. 合并训练集和测试集的点击记录

    Parameters
    ----------
    data_path : Path
        数据路径
    df_article : pd.DataFrame
        文章特征表

    Returns
    -------
    pd.DataFrame
        用户点击历史记录，按user_id和click_timestamp排序
    """
    # 加载原始点击数据
    df_train_click = pd.read_csv(data_path / 'train_click_log.csv')
    df_test_click = pd.read_csv(data_path / 'testA_click_log.csv')

    # 按用户和时间排序
    df_train_click = df_train_click.sort_values(
        by=['user_id', 'click_timestamp']
    ).reset_index(drop=True)

    # 训练集：排除每个用户的最后一次点击（留出作为验证标签）
    # 使用apply对每个用户组取除最后一个外的所有点击
    df_train_click = df_train_click.groupby('user_id').apply(
        lambda x: x.iloc[:-1]
    ).reset_index(drop=True)

    # 合并训练集（排除last1）和测试集（全量）
    df_click = pd.concat([df_train_click, df_test_click], axis=0)

    # 按用户和时间排序
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)

    # 重命名列名统一（click_article_id -> article_id）
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)

    # 合并文章特征到点击记录
    df_click = df_click.merge(df_article, how='left')

    # 时间戳从毫秒转换为秒
    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000

    # 转换时间戳为datetime格式，便于提取时间特征
    df_click['click_datetime'] = pd.to_datetime(
        df_click['click_timestamp'], unit='s', errors='coerce'
    )

    # 提取点击小时特征（用于分析用户活跃时间段）
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    return df_click


def build_time_diff_features(df_click):
    """
    构建时间差相关特征

    特征列表：
    1. user_id_click_article_created_at_ts_diff_mean: 用户点击文章的创建时间差的平均值
       （反映用户偏好新文章还是旧文章）
    2. user_id_click_diff_mean: 用户点击时间间隔的平均值
       （反映用户的活跃频率）
    3. user_click_timestamp_created_at_ts_diff_mean/std: 点击时间与文章创建时间差的均值/标准差
       （反映用户点击文章时的"新鲜度"偏好）

    Parameters
    ----------
    df_click : pd.DataFrame
        用户点击历史

    Returns
    -------
    pd.DataFrame
        用户级别的时间差特征
    """
    features = pd.DataFrame({'user_id': df_click['user_id'].unique()})

    # 1. 用户点击文章的创建时间差的平均值
    # 计算同一用户连续点击的文章创建时间差
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby('user_id')['created_at_ts'].diff()
    temp = df_click.groupby('user_id')['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    temp.columns = ['user_id', 'user_id_click_article_created_at_ts_diff_mean']
    features = features.merge(temp, on='user_id', how='left')

    # 2. 用户点击时间差的平均值
    # 计算同一用户连续点击的时间间隔
    df_click['user_id_click_diff'] = df_click.groupby('user_id')['click_timestamp'].diff()
    temp = df_click.groupby('user_id')['user_id_click_diff'].mean().reset_index()
    temp.columns = ['user_id', 'user_id_click_diff_mean']
    features = features.merge(temp, on='user_id', how='left')

    # 3. 点击时间与文章创建时间之差的统计值
    # 这个值越小，说明用户越喜欢看新发布的文章
    df_click['click_timestamp_created_at_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']
    temp = df_click.groupby('user_id').agg(
        user_click_timestamp_created_at_ts_diff_mean=('click_timestamp_created_at_ts_diff', 'mean'),
        user_click_timestamp_created_at_ts_diff_std=('click_timestamp_created_at_ts_diff', 'std')
    ).reset_index()
    features = features.merge(temp, on='user_id', how='left')

    return features


def build_user_behavior_features(df_click):
    """
    构建用户行为统计特征

    特征列表：
    1. user_click_datetime_hour_std: 点击时间小时的标准差
       （反映用户活跃时间的集中程度）
    2. user_clicked_article_words_count_mean: 点击文章字数的平均值
       （反映用户偏好长文还是短文）
    3. user_click_last_article_words_count: 最后一次点击文章的字数
    4. user_click_last_article_created_time: 最后一次点击文章的创建时间
    5. user_clicked_article_created_time_max: 点击文章的最大创建时间
    6. user_click_last_article_click_time: 最后一次点击时间
    7. user_clicked_article_click_time_mean: 平均点击时间

    Parameters
    ----------
    df_click : pd.DataFrame
        用户点击历史

    Returns
    -------
    pd.DataFrame
        用户行为统计特征
    """
    features = pd.DataFrame({'user_id': df_click['user_id'].unique()})

    # 1. 点击时间小时的统计（活跃时间段分布）
    temp = df_click.groupby('user_id').agg(
        user_click_datetime_hour_std=('click_datetime_hour', 'std')
    ).reset_index()
    features = features.merge(temp, on='user_id', how='left')

    # 2. 点击文章字数的统计
    temp = df_click.groupby('user_id').agg(
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', lambda x: x.iloc[-1])  # 最后一个点击的文章字数
    ).reset_index()
    features = features.merge(temp, on='user_id', how='left')

    # 3. 点击文章的创建时间统计
    temp = df_click.groupby('user_id').agg(
        user_click_last_article_created_time=('created_at_ts', lambda x: x.iloc[-1]),
        user_clicked_article_created_time_max=('created_at_ts', 'max')
    ).reset_index()
    features = features.merge(temp, on='user_id', how='left')

    # 4. 点击时间的统计
    temp = df_click.groupby('user_id').agg(
        user_click_last_article_click_time=('click_timestamp', lambda x: x.iloc[-1]),
        user_clicked_article_click_time_mean=('click_timestamp', 'mean')
    ).reset_index()
    features = features.merge(temp, on='user_id', how='left')

    return features


def build_cross_features(df_feature):
    """
    构建候选文章与用户历史行为的交叉特征

    特征列表：
    1. user_last_click_created_at_ts_diff: 候选文章创建时间 - 用户最后点击文章创建时间
    2. user_last_click_timestamp_diff: 候选文章创建时间 - 用户最后点击时间
    3. user_last_click_words_count_diff: 候选文章字数 - 用户最后点击文章字数

    这些特征反映候选文章与用户历史阅读习惯的匹配程度

    Parameters
    ----------
    df_feature : pd.DataFrame
        基础特征表

    Returns
    -------
    pd.DataFrame
        添加了交叉特征的DataFrame
    """
    # 候选文章与用户最后点击文章的创建时间差
    df_feature['user_last_click_created_at_ts_diff'] = (
        df_feature['created_at_ts'] - df_feature['user_click_last_article_created_time']
    )

    # 候选文章创建时间与用户最后点击时间的差
    df_feature['user_last_click_timestamp_diff'] = (
        df_feature['created_at_ts'] - df_feature['user_click_last_article_click_time']
    )

    # 候选文章字数与用户最后点击文章字数的差
    df_feature['user_last_click_words_count_diff'] = (
        df_feature['words_count'] - df_feature['user_click_last_article_words_count']
    )

    return df_feature


def build_count_features(df_click):
    """
    构建计数统计特征

    特征列表：
    1. user_id_cnt: 用户点击次数（用户活跃度）
    2. article_id_cnt: 文章被点击次数（文章热度）
    3. user_id_category_id_cnt: 用户在每个类别的点击次数（用户类别偏好强度）

    Parameters
    ----------
    df_click : pd.DataFrame
        用户点击历史

    Returns
    -------
    pd.DataFrame
        计数特征，需要与主表merge
    """
    # 定义分组组合
    group_combinations = [
        ['user_id'],                    # 用户点击次数
        ['article_id'],                 # 文章被点击次数
        ['user_id', 'category_id']      # 用户在每个类别的点击次数
    ]

    count_features = pd.DataFrame({'user_id': df_click['user_id'].unique()})

    for group_cols in group_combinations:
        df_temp = df_click.groupby(group_cols).size().reset_index()
        feature_name = '{}_cnt'.format('_'.join(group_cols))
        df_temp.columns = group_cols + [feature_name]

        # 对于多列分组，需要保留所有分组列用于merge
        if len(group_cols) == 1:
            count_features = count_features.merge(df_temp, on=group_cols[0], how='left')
        else:
            # 多列分组特征需要直接merge到主表
            pass

    return count_features


def build_all_count_features(df_click, df_feature):
    """
    构建所有计数特征并合并到主表

    Parameters
    ----------
    df_click : pd.DataFrame
        用户点击历史
    df_feature : pd.DataFrame
        主特征表

    Returns
    -------
    pd.DataFrame
        添加了计数特征的主表
    """
    # 用户点击次数
    temp = df_click.groupby('user_id').size().reset_index()
    temp.columns = ['user_id', 'user_id_cnt']
    df_feature = df_feature.merge(temp, on='user_id', how='left')

    # 文章被点击次数（文章热度）
    temp = df_click.groupby('article_id').size().reset_index()
    temp.columns = ['article_id', 'article_id_cnt']
    df_feature = df_feature.merge(temp, on='article_id', how='left')

    # 用户在每个类别的点击次数
    temp = df_click.groupby(['user_id', 'category_id']).size().reset_index()
    temp.columns = ['user_id', 'category_id', 'user_id_category_id_cnt']
    df_feature = df_feature.merge(temp, on=['user_id', 'category_id'], how='left')

    return df_feature


def main():
    """主函数"""
    # ==================== 1. 初始化 ====================
    log = setup_logger()
    log.info("=" * 60)
    log.info("特征工程模块启动")
    log.info("=" * 60)

    # ==================== 2. 加载召回候选 ====================
    log.info("【步骤1】加载召回候选结果...")
    df_train, df_tst, df_feature = load_recall_candidates(RECALL_PATH, RECALL_NUM)
    log.info(f"训练集候选数: {len(df_train)}, 测试集候选数: {len(df_tst)}")

    # ==================== 3. 加载文章特征 ====================
    log.info("【步骤2】加载文章基础特征...")
    df_article = load_article_features(DATA_PATH)

    # 合并文章特征到候选集
    df_feature = df_feature.merge(df_article, how='left')

    # 转换文章创建时间为datetime格式（便于后续特征工程）
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s'
    )

    log.info(f"合并文章特征后: {df_feature.shape}")

    # ==================== 4. 构建点击历史 ====================
    log.info("【步骤3】构建用户点击历史...")
    df_click = build_click_history(DATA_PATH, df_article)
    log.info(f"点击历史记录数: {len(df_click)}")

    # ==================== 5. 构建时间差特征 ====================
    log.info("【步骤4】构建时间差特征...")
    time_features = build_time_diff_features(df_click)
    df_feature = df_feature.merge(time_features, on='user_id', how='left')
    log.info(f"合并时间差特征后: {df_feature.shape}")

    # ==================== 6. 构建用户行为特征 ====================
    log.info("【步骤5】构建用户行为统计特征...")
    behavior_features = build_user_behavior_features(df_click)
    df_feature = df_feature.merge(behavior_features, on='user_id', how='left')
    log.info(f"合并行为特征后: {df_feature.shape}")

    # ==================== 7. 构建交叉特征 ====================
    log.info("【步骤6】构建交叉特征...")
    df_feature = build_cross_features(df_feature)
    log.info(f"构建交叉特征后: {df_feature.shape}")

    # ==================== 8. 构建计数特征 ====================
    log.info("【步骤7】构建计数统计特征...")
    df_feature = build_all_count_features(df_click, df_feature)
    log.info(f"合并计数特征后: {df_feature.shape}")

    # ==================== 9. 保存特征文件 ====================
    log.info("【步骤8】保存特征文件...")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    output_file = SAVE_PATH / f'df_feature_{RECALL_NUM}.csv'
    df_feature.to_csv(output_file, index=False)
    log.info(f"特征文件已保存: {output_file}")

    log.info("=" * 60)
    log.info("特征工程完成")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
