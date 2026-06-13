"""
ItemCF (Item Collaborative Filtering) 推荐系统主程序
基于物品的协同过滤算法，通过计算物品之间的相似度来进行推荐
"""

import os
import sys
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.utils import Logger, gen_user_item_time
from Recall_Methods import evaluate, itemcf_sim_cal_parallel, I2I_recall_parallel

# 配置
warnings.filterwarnings('ignore')
random.seed(3)


# ==================== 路径配置 ====================
# 相对于 Code/Recall 目录的路径
# Code/Recall/ -> ../ -> Code/ -> Initial_data/
DATA_PATH = Path('../Initial_data')
SIM_MATRIX_PATH = Path('../Results/sim_matrix')   #相似度矩阵
RECALL_DICT_PATH = Path('../Results/Recall_dict')   #召回结果
SUBMIT_PATH = Path('../Results/submit')      #提交结果
LOG_PATH = Path('../Results/log')

TRAIN_DATA_PATH = DATA_PATH / 'train_click_log.csv'
TEST_DATA_PATH = DATA_PATH / 'testA_click_log.csv'
ITEM_INFO_PATH = DATA_PATH / 'articles.csv'


def setup_logger():
    """初始化日志记录器"""
    now = datetime.now()
    formatted_date = now.strftime("%Y%m%d")
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    log_file = str(LOG_PATH / f'test_itemcf_{formatted_date}.txt')
    return Logger(log_file).logger


def load_data():
    """
    加载训练数据、测试数据和物品信息

    Returns:
        tuple: (train_data, test_data, item_info, all_data)
    """
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    item_info = pd.read_csv(ITEM_INFO_PATH)
    item_info = item_info.rename(columns={'article_id': 'click_article_id'})

    # 合并训练集和测试集作为全量数据
    all_data = pd.concat([train_data, test_data], axis=0)

    return train_data, test_data, item_info, all_data


def split_train_val(all_data):
    """
    将用户行为数据划分为训练历史序列和验证标签

    划分策略：对每个用户，按时间升序排列，取前n-1个作为训练历史，最后1个作为验证标签

    Args:
        all_data: 全量点击数据

    Returns:
        tuple: (train_hist_df, train_last_df)
            - train_hist_df: 用户历史点击序列（除最后一次外）
            - train_last_df: 用户最后一次点击（作为验证标签）
    """
    # 过滤掉只有一个点击记录的用户（无法划分训练集和验证集）
    df_filtered = all_data.groupby('user_id').filter(lambda x: len(x) > 1).reset_index(drop=True)

    # 按用户ID和时间戳升序排序
    df_filtered = df_filtered.sort_values(by=['user_id', 'click_timestamp'], ascending=True)

    # 取每个用户的前n-1个点击作为训练历史
    train_hist_df = df_filtered.groupby('user_id').apply(
        lambda x: x.iloc[:-1]
    ).reset_index(drop=True)

    # 取每个用户的最后一次点击作为验证标签
    train_last_df = df_filtered.groupby('user_id').tail(1).reset_index(drop=True)

    return train_hist_df, train_last_df


def build_item_time_dict(item_info):
    """
    构建物品创建时间归一化字典（用于后续相似度计算的时间加权）

    Args:
        item_info: 物品信息DataFrame

    Returns:
        dict: {item_id: normalized_created_time}
    """
    min_max_scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    item_creat_ts_norm = item_info[['click_article_id', 'created_at_ts']].copy()
    item_creat_ts_norm['created_at_ts'] = item_creat_ts_norm[['created_at_ts']].apply(min_max_scale)

    return dict(zip(item_creat_ts_norm['click_article_id'], item_creat_ts_norm['created_at_ts']))


def get_hot_items(all_data, top_k=350):
    """
    获取热门物品列表（用于冷启动补全）

    Args:
        all_data: 全量点击数据
        top_k: 取前k个热门物品

    Returns:
        Index: 热门物品ID列表
    """
    return all_data['click_article_id'].value_counts().index[:top_k]


def build_and_save_sim_matrix(click_seq_dict, item_time_dict, save_path, log):
    """
    构建物品相似度矩阵并保存

    Args:
        click_seq_dict: 用户点击序列字典 {user_id: [(item_id, timestamp), ...]}
        item_time_dict: 物品创建时间归一化字典
        save_path: 相似度矩阵保存路径
        log: 日志记录器

    Returns:
        dict: 物品相似度矩阵 {item_id: {related_item_id: sim_score, ...}, ...}
    """
    # 如果文件已存在且不需要重新计算，可以直接加载
    if save_path.exists():
        log.info(f"相似度矩阵已存在，从 {save_path} 加载")
        return pickle.load(open(save_path, 'rb'))

    # 并行计算物品相似度矩阵
    log.info("开始构建物品相似度矩阵...")
    itemcf_sim_cal_parallel(click_seq_dict, item_time_dict, save_path, log)

    return pickle.load(open(save_path, 'rb'))


def log_sim_matrix_stats(sim_matrix, log):
    """
    记录相似度矩阵的统计信息

    Args:
        sim_matrix: 物品相似度矩阵
        log: 日志记录器
    """
    sim_len = [len(sim_items) for sim_items in sim_matrix.values()]
    log.info(f'相似度矩阵统计: max={np.max(sim_len)}, min={np.min(sim_len)}, mean={np.mean(sim_len):.2f}')


def evaluate_recall_results(recall_dict_path, recall_file, train_last_df, log):
    """
    在验证集上评估召回效果

    Args:
        recall_dict_path: 召回结果保存路径
        recall_file: 召回结果文件名
        train_last_df: 验证集标签（用户最后一次点击）
        log: 日志记录器
    """
    # 加载召回结果
    df_data = pickle.load(open(recall_dict_path / recall_file, 'rb'))
    df_data = df_data.sort_values(
        ['user_id', 'sim_score'], ascending=[True, False]
    ).reset_index(drop=True)

    # 只评估训练集部分（user_id < 200000）
    df_data_train = df_data[df_data['user_id'] < 200000]
    total = train_last_df[train_last_df['user_id'] < 200000]['user_id'].nunique()

    # 计算评估指标 (HitRate@K, MRR@K)
    metrics = evaluate(df_data_train[df_data_train['label'].notnull()], total, 'article_id')
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = metrics

    # 记录评估结果
    log.info(f'\n===== 验证集评估结果 ({recall_file}) =====')
    log.info(f'HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}')
    log.info(f'HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}')
    log.info(f'HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}')
    log.info(f'HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}')
    log.info(f'HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}')


def generate_submission(recall_dict_path, recall_file, submit_file):
    """
    生成提交文件
    将每个用户的top5推荐结果展开为wide格式（一行一个用户）

    Args:
        recall_dict_path: 召回结果保存路径
        recall_file: 召回结果文件名
        submit_file: 提交文件保存路径
    """
    # 加载测试集召回结果
    df_data = pickle.load(open(recall_dict_path / recall_file, 'rb'))
    df_data['article_id'] = df_data['article_id'].astype(int)
    df_data = df_data.sort_values(
        ['user_id', 'sim_score'], ascending=[True, False]
    ).reset_index(drop=True)

    # 取每个用户的top5推荐
    df_top5 = df_data.groupby('user_id').head(5).reset_index(drop=True)

    # 为每个用户的推荐结果添加排名
    df_top5['rank'] = df_top5.groupby('user_id')['sim_score'].rank(
        method='first', ascending=False
    )

    # 将长格式转换为宽格式（每个用户的top5展开为一行）
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

    print(df_pivot)

    # 保存提交文件
    SUBMIT_PATH.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(submit_file, index=False, encoding='utf-8-sig')
    print(f"提交文件已保存至: {submit_file}")


def main():
    """主函数"""
    # ==================== 1. 初始化 ====================
    log = setup_logger()
    log.info("=" * 60)
    log.info("ItemCF 推荐系统启动")
    log.info("=" * 60)

    # ==================== 2. 数据加载与预处理 ====================
    log.info("【步骤1】加载数据...")
    train_data, test_data, item_info, all_data = load_data()

    # 划分训练集（历史行为）和验证集（最后一次点击）
    train_hist_df, train_last_df = split_train_val(all_data)

    # 构建辅助数据
    item_creat_ts_norm_dict = build_item_time_dict(item_info)
    hot_items = get_hot_items(all_data)

    log.info(f"  训练集用户数: {train_hist_df['user_id'].nunique()}")
    log.info(f"  验证集标签数: {len(train_last_df)}")
    log.info(f"  测试集用户数: {test_data['user_id'].nunique()}")

    # ==================== 3. 验证阶段：构建相似度矩阵并评估 ====================
    log.info("\n" + "=" * 60)
    log.info("【阶段1】验证集评估")
    log.info("=" * 60)

    # 构建用户历史点击序列字典（用于计算相似度矩阵）
    click_seq_dict = gen_user_item_time(
        train_hist_df[['user_id', 'click_article_id', 'click_timestamp']]
    )

    # 计算物品相似度矩阵
    sim_matrix_file = SIM_MATRIX_PATH / 'itemcf_sim.pkl'
    SIM_MATRIX_PATH.mkdir(parents=True, exist_ok=True)
    itemcf_sim_matrix = build_and_save_sim_matrix(
        click_seq_dict, item_creat_ts_norm_dict, sim_matrix_file, log
    )

    # 记录相似度矩阵统计信息
    log_sim_matrix_stats(itemcf_sim_matrix, log)


    # 在验证集上进行召回
    RECALL_DICT_PATH.mkdir(parents=True, exist_ok=True)
    I2I_recall_parallel(
        i2i_matrix=itemcf_sim_matrix,   # 相似度矩阵（刚算好的）
        hist_df=train_hist_df,   # 用户历史行为
        item_creat_ts_norm_dict=item_creat_ts_norm_dict,  # 物品时间字典
        hot_item=hot_items,
        method='sim',   # 方法标识
        sim_item_topk=226,  #每个物品找226个相似物品
        recall_item_num=100, # 每个用户召回100个
               recall_dict_path=RECALL_DICT_PATH,
        last_df=train_last_df
    )

    # 评估验证集效果
    evaluate_recall_results(RECALL_DICT_PATH, 'sim_recall.pkl', train_last_df, log)

    # ==================== 4. 测试集预测阶段 ====================
    log.info("\n" + "=" * 60)
    log.info("【阶段2】测试集预测")
    log.info("=" * 60)

    # 使用全量数据重新构建相似度矩阵（包含更完整的用户行为信息）
    full_hist_df = all_data[['user_id', 'click_article_id', 'click_timestamp']]
    full_click_seq_dict = gen_user_item_time(full_hist_df)

    sim_matrix_file_tst = SIM_MATRIX_PATH / 'itemcf_sim_tst.pkl'
    itemcf_sim_matrix_tst = build_and_save_sim_matrix(
        full_click_seq_dict, item_creat_ts_norm_dict, sim_matrix_file_tst, log
    )

    # 在测试集上进行召回
    I2I_recall_parallel(
        i2i_matrix=itemcf_sim_matrix_tst,
        hist_df=test_data,
        item_creat_ts_norm_dict=item_creat_ts_norm_dict,
        hot_item=hot_items,
        method='sim_tst',
        sim_item_topk=200,
        recall_item_num=100,
        recall_dict_path=RECALL_DICT_PATH
    )

    # 生成提交文件
    submit_file = SUBMIT_PATH / 'itemcf_recom.csv'
    generate_submission(RECALL_DICT_PATH, 'sim_tst_recall.pkl', submit_file)

    log.info("\n" + "=" * 60)
    log.info("ItemCF 推荐系统运行完成")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
