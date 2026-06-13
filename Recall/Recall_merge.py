"""
多路召回结果融合模块

功能：将多种召回方法（如ItemCF、DSSM等）的结果进行融合，生成最终的推荐列表
融合策略：投票融合 + 加权归一化

处理流程：
1. 加载各路召回结果（训练集和测试集）
2. 使用投票融合策略合并多路召回
3. 对不足topk的用热门物品补全
4. 生成提交文件
"""

import pickle
import pandas as pd
from pathlib import Path
from Recall_Methods import evaluate


# ==================== 路径配置 ====================
# 相对于 Code/Recall 目录的路径
DATA_PATH = Path('../Initial_data')
RECALL_DICT_PATH = Path('../Results/Recall_dict')
SUBMIT_PATH = Path('../Results/submit')

TRAIN_DATA_PATH = DATA_PATH / 'train_click_log.csv'
TEST_DATA_PATH = DATA_PATH / 'testA_click_log.csv'
ITEM_INFO_PATH = DATA_PATH / 'articles.csv'


# ==================== 融合策略配置 ====================
# 排序策略选择
# 0: 按投票数(n_votes)优先排序（多路共同召回的排前面）
# 1: 按融合分数(sim_score)排序（加权分数高的排前面）
STRATEGY = 0

# 各路召回权重配置（可配置多组权重进行实验）
RECALL_WEIGHTS_CONFIGS = [
    {'itemcf': 0.5, 'dssm': 0.5},   # 均等权重
    # {'itemcf': 0.4, 'dssm': 0.6},   # DSSM权重更高
    {'itemcf': 0.45, 'dssm': 0.55},
    # {'itemcf': 0.35, 'dssm': 0.65},
]

# TopK设置（最终每个用户保留多少候选物品）
TOP_K = 50


def voting_fusion(user_multi_recall_dict, recall_weights):
    """
    使用「投票融合 + 加权归一化」方法，合并多路召回结果。

    核心思想：
    1. 归一化：对不同召回方法的分数进行min-max归一化（消除量纲差异）
    2. 加权：根据配置的权重对归一化后的分数加权
    3. 投票：统计有多少路召回了同一个物品（n_votes）
    4. 聚合：对同一(user, item)的多路分数进行累加

    Parameters
    ----------
    user_multi_recall_dict : dict[str, pd.DataFrame]
        多路召回结果字典，key为召回方法名称(如'itemcf', 'dssm')，
        value为对应的DataFrame，必须包含列：user_id, article_id, sim_score

    recall_weights : dict[str, float]
        每种召回方法的权重。若某通道未配置，默认权重=1.0

    Returns
    -------
    pd.DataFrame
        融合后的结果，包含列：
        - user_id: 用户ID
        - article_id: 文章ID
        - label: 标签（训练集有，测试集为-1）
        - n_votes: 该物品被多少路召回共同召回
        - sim_score: 加权归一化后的融合总分
        - {method}_score: 各路原始的相似度分数
    """
    df_list = []
    all_methods = list(user_multi_recall_dict.keys())

    for method_name, df_recall in user_multi_recall_dict.items():
        temp = df_recall.copy(deep=True)

        # 保留原始分数用于后续分析
        temp[f'{method_name}_score'] = temp['sim_score']

        # 1) 计算该通道的全局 min/max
        min_v = temp['sim_score'].min()
        max_v = temp['sim_score'].max()
        print(f'{method_name}: min={min_v:.4f}, max={max_v:.4f}')

        # 2) Min-Max归一化到[0.1, 1.0]范围
        #    避免0分，保留一定区分度
        if max_v != min_v:
            temp['sim_score'] = 0.1 + (temp['sim_score'] - min_v) / (max_v - min_v) * 0.9
        else:
            temp['sim_score'] = 0.0

        # 3) 加权
        weight = recall_weights.get(method_name, 1.0)
        temp['sim_score'] = temp['sim_score'] * weight

        # 4) 标记来源方法（用于投票计数）
        temp['method_name'] = method_name

        df_list.append(temp)

    # 5) 合并所有通道
    merged_all = pd.concat(df_list, ignore_index=True)

    # 6) 按(user, article, label)聚合
    #    - n_votes: 统计有多少路召回了这个物品
    #    - sim_score: 各路加权分数求和
    group_cols = ['user_id', 'article_id', 'label']

    agg_dict = {
        'n_votes': ('method_name', 'nunique'),
        'sim_score': ('sim_score', 'sum')
    }

    # 同时保留各路原始分数
    for method in all_methods:
        col_name = f'{method}_score'
        if col_name in merged_all.columns:
            agg_dict[col_name] = (col_name, 'max')

    fused_df = (
        merged_all
        .groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    # 填充缺失值为0（某路没有召回该物品）
    for method in all_methods:
        col_name = f'{method}_score'
        fused_df[col_name] = fused_df[col_name].fillna(0)

    return fused_df


def fill_hot_items(group, hot_item, topk):
    """
    对不足topk的用户，用热门物品补全

    Parameters
    ----------
    group : pd.DataFrame
        单个用户的推荐列表
    hot_item : list
        热门物品列表
    topk : int
        目标数量

    Returns
    -------
    pd.DataFrame
        补全后的推荐列表
    """
    user_id = group['user_id'].iloc[0]
    current_articles = group['article_id'].tolist()
    current_len = len(group)

    # 如果当前推荐数量不足topk，用热门文章补全
    if current_len < topk:
        need_num = topk - current_len
        fill_items = []

        for item in hot_item:
            if item not in current_articles:
                fill_items.append({
                    'user_id': user_id,
                    'article_id': item,
                    'sim_score': 0,
                    'itemcf_score': 0,
                    'dssm_score': 0
                })
                need_num -= 1
                if need_num <= 0:
                    break

        if fill_items:
            fill_df = pd.DataFrame(fill_items)
            group = pd.concat([group, fill_df], ignore_index=True)

    return group


def load_recall_results(recall_dict_path):
    """
    加载各路召回结果

    Returns:
        tuple: (itemcf_recall, dssm_recall, itemcf_recall_tst, dssm_recall_tst)
    """
    # 测试集召回结果（用于最终提交）
    itemcf_recall_tst = pickle.load(open(recall_dict_path / 'sim_tst_recall.pkl', 'rb'))
    dssm_recall_tst = pickle.load(open(recall_dict_path / 'dssm_tst_recall.pkl', 'rb'))

    print("ItemCF测试集召回:", itemcf_recall_tst.shape)
    print("DSSM测试集召回:", dssm_recall_tst.shape)

    # 训练集召回结果（用于离线验证，只保留user_id<200000的训练用户）
    itemcf_recall = pickle.load(open(recall_dict_path / 'sim_recall.pkl', 'rb'))
    itemcf_recall = itemcf_recall[itemcf_recall['user_id'] < 200000]

    dssm_recall = pickle.load(open(recall_dict_path / 'dssm_recall.pkl', 'rb'))
    dssm_recall = dssm_recall[dssm_recall['user_id'] < 200000]

    print("ItemCF训练集召回:", itemcf_recall.shape)
    print("DSSM训练集召回:", dssm_recall.shape)

    return itemcf_recall, dssm_recall, itemcf_recall_tst, dssm_recall_tst


def prepare_validation_data():
    """
    准备验证所需的数据（热门物品、验证集标签）

    Returns:
        tuple: (hot_item, trn_last_df)
    """
    # 加载数据
    trn_data = pd.read_csv(TRAIN_DATA_PATH)
    tst_data = pd.read_csv(TEST_DATA_PATH)
    item_info = pd.read_csv(ITEM_INFO_PATH)

    item_info = item_info.rename(columns={'article_id': 'click_article_id'})

    # 合并全量数据
    all_data = pd.concat([trn_data, tst_data], axis=0)
    all_data = all_data.merge(item_info, how='left', on='click_article_id')
    all_data = all_data.sort_values(by=['user_id', 'click_timestamp']).reset_index(drop=True)

    # 获取热门物品（用于冷启动补全）
    hot_item = all_data['click_article_id'].value_counts().index[:350]

    # 提取验证集标签（每个用户的最后一次点击）
    df_filter = all_data.groupby('user_id').filter(lambda x: len(x) > 1).reset_index(drop=True)
    df_filter = df_filter.sort_values(by=['user_id', 'click_timestamp'], ascending=True)
    trn_last_df = df_filter.groupby('user_id').tail(1).reset_index(drop=True)

    return hot_item, trn_last_df


def evaluate_train_set(df_data, trn_last_df, strategy):
    """
    在训练集上评估融合效果

    Parameters
    ----------
    df_data : pd.DataFrame
        融合后的数据
    trn_last_df : pd.DataFrame
        验证集标签
    strategy : int
        排序策略
    """
    # 只评估训练集部分
    total = trn_last_df[trn_last_df['user_id'] < 200000]['user_id'].nunique()

    # 根据策略排序（必须在筛选训练集之前排序！）
    if strategy == 0:
        # 策略0：先按投票数排序（多路共同召回的优先）
        df_data = df_data.sort_values(
            by=['user_id', 'n_votes', 'sim_score'],
            ascending=[True, False, False]
        )
    elif strategy == 1:
        # 策略1：直接按融合分数排序
        df_data = df_data.sort_values(
            by=['user_id', 'sim_score'],
            ascending=[True, False]
        ).reset_index(drop=True)

    # 排序后再筛选训练集部分
    df_data_train = df_data[df_data['user_id'] < 200000]

    # 计算评估指标
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_data_train[df_data_train['label'].notnull()], total, 'article_id'
    )

    print(f'\n===== 融合评估结果（策略{strategy}） =====')
    print(f'HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}')
    print(f'HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}')
    print(f'HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}')
    print(f'HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}')
    print(f'HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}')

    return df_data


def save_train_results(df_data, recall_dict_path, topk):
    """
    保存训练集融合结果

    Parameters
    ----------
    df_data : pd.DataFrame
        融合后的数据
    recall_dict_path : Path
        保存路径
    topk : int
        保留topk个候选
    """
    cols = ['user_id', 'article_id', 'sim_score', 'itemcf_score', 'dssm_score', 'label']
    df_data_train = df_data[df_data['user_id'] < 200000][cols]
    df_data_train = df_data_train.sort_values(by=['user_id', 'sim_score'], ascending=[True, False])

    # 取topk保存
    final_df_topk = df_data_train.groupby('user_id', group_keys=False).head(topk)
    pickle.dump(final_df_topk, open(recall_dict_path / f'train_final_recall_{topk}.pkl', 'wb'))

    # 统计没有正样本的用户数（用于检查召回质量）
    user_positive_flag = final_df_topk.groupby('user_id')['label'].apply(lambda x: (x == 1).any())
    n_users_without_positive = (~user_positive_flag).sum()
    print(f"训练集中没有正样本的用户数：{n_users_without_positive}")

    return final_df_topk


def save_test_results(df_data, recall_dict_path, submit_path, strategy, topk):
    """
    保存测试集结果并生成提交文件

    Parameters
    ----------
    df_data : pd.DataFrame
        融合后的数据
    recall_dict_path : Path
        召回结果保存路径
    submit_path : Path
        提交文件保存路径
    strategy : int
        策略编号（用于文件名）
    topk : int
        保留topk个候选
    """
    # 只保留测试集用户
    df_data_test = df_data[df_data['user_id'] >= 200000]

    # 选择需要的列并排序
    cols = ['user_id', 'article_id', 'sim_score', 'itemcf_score', 'dssm_score']
    df_data_test = df_data_test[cols].sort_values(
        by=['user_id', 'sim_score'],
        ascending=[True, False]
    )

    # 保存topk候选
    final_df_topk = df_data_test.groupby('user_id', group_keys=False).head(topk)

    # 过滤有效测试用户（200000-249999）
    final_df_topk = final_df_topk[
        (final_df_topk['user_id'] >= 200000) &
        (final_df_topk['user_id'] <= 249999)
    ].reset_index(drop=True)

    pickle.dump(final_df_topk, open(recall_dict_path / f'tst_final_recall_{topk}.pkl', 'wb'))

    # 生成提交文件（取top5）
    df_top5 = df_data_test.groupby('user_id').head(5).reset_index(drop=True)
    df_top5['rank'] = df_top5.groupby('user_id')['sim_score'].rank(method='first', ascending=False)

    # 将长格式转换为宽格式（每个用户的top5展开为一行）
    df_pivot = df_top5.pivot(index='user_id', columns='rank', values='article_id')
    df_pivot.columns = df_pivot.columns.astype(int)
    df_pivot = df_pivot.rename(columns={
        1: 'article_1', 2: 'article_2', 3: 'article_3',
        4: 'article_4', 5: 'article_5'
    })
    df_pivot = df_pivot.reset_index()

    # 过滤有效测试用户
    df_pivot = df_pivot[
        (df_pivot['user_id'] >= 200000) &
        (df_pivot['user_id'] <= 249999)
    ].reset_index(drop=True)

    print("\n提交文件预览:")
    print(df_pivot)

    # 保存提交文件
    submit_path.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(submit_path / f'merge_recom_{strategy}.csv', index=False, encoding='utf-8-sig')
    print(f"\n提交文件已保存: {submit_path / f'merge_recom_{strategy}.csv'}")


def main():
    """主函数"""
    print("=" * 60)
    print("多路召回融合模块启动")
    print("=" * 60)

    # ==================== 1. 加载召回结果 ====================
    print("\n【步骤1】加载各路召回结果...")
    itemcf_recall, dssm_recall, itemcf_recall_tst, dssm_recall_tst = load_recall_results(RECALL_DICT_PATH)

    # 合并训练集和测试集（用于后续统一处理）
    user_multi_recall_dict = {
        'itemcf': pd.concat([itemcf_recall, itemcf_recall_tst], axis=0),
        'dssm': pd.concat([dssm_recall, dssm_recall_tst], axis=0)
    }

    # ==================== 2. 准备验证数据 ====================
    print("\n【步骤2】准备验证数据...")
    hot_item, trn_last_df = prepare_validation_data()
    print(f"热门物品数: {len(hot_item)}")

    # ==================== 3. 多路融合 ====================
    print("\n【步骤3】进行多路召回融合...")

    for recall_weights in RECALL_WEIGHTS_CONFIGS:
        print(f"\n当前权重配置: {recall_weights}")

        # 融合
        final_df = voting_fusion(user_multi_recall_dict, recall_weights)
        print(f"融合后候选数: {len(final_df)}")

        # 热门补全
        df_data = final_df.groupby('user_id').apply(
            lambda x: fill_hot_items(x, hot_item, TOP_K)
        ).reset_index(drop=True)

        # ==================== 4. 训练集评估 ====================
        print("\n【步骤4】训练集评估...")
        df_data = evaluate_train_set(df_data, trn_last_df, STRATEGY)

        # 保存训练集结果
        save_train_results(df_data, RECALL_DICT_PATH, TOP_K)

        # ==================== 5. 测试集结果生成 ====================
        print("\n【步骤5】生成测试集提交文件...")
        save_test_results(df_data, RECALL_DICT_PATH, SUBMIT_PATH, STRATEGY, TOP_K)

    print("\n" + "=" * 60)
    print("多路召回融合完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
