"""
DSSM (Deep Structured Semantic Model) 召回模型
基于深度学习的双塔模型，分别学习用户和物品的向量表示，通过向量相似度进行召回
"""

import os
import sys
import math
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from deepmatch.models import DSSM
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.utils import Logger, gen_user_item_time
from Recall_Methods import evaluate
from recall_data_gen import build_train_test_data, build_infer_data, gen_infer_submit


# ==================== 训练配置 ====================
# 训练数据选择
# True:  使用全量数据训练（用于最终提交，无法进行离线验证）
# False: 使用去掉最后一次点击的数据训练（留出最后一次用于离线验证）
USE_FULL_DATA_FOR_TRAINING = False

# 快速测试模式（只用少量数据验证代码逻辑）
# True:  只使用100个用户的数据快速测试
# False: 使用全部数据（正式运行）
DEBUG_MODE = False


# ==================== 路径配置 ====================
# 相对于 Code/Recall 目录的路径
DATA_PATH = Path('../Initial_data')
MODEL_PATH = Path('../Results/Model')
RECALL_DICT_PATH = Path('../Results/Recall_dict')
SUBMIT_PATH = Path('../Results/submit')
LOG_PATH = Path('../Results/log')
SAVE_DIR = Path('../Results/DSSM')

TRAIN_DATA_PATH = DATA_PATH / 'train_click_log.csv'
TEST_DATA_PATH = DATA_PATH / 'testA_click_log.csv'
ITEM_EMB_PATH = DATA_PATH / 'articles_emb.csv'
ITEM_INFO_PATH = DATA_PATH / 'articles.csv'


def setup_logger():
    """初始化日志记录器"""
    now = datetime.now()
    formatted_date = now.strftime("%Y%m%d")
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    current_file_name = Path(__file__).stem
    log_file = str(LOG_PATH / f'test_{current_file_name}_{formatted_date}.txt')
    return Logger(log_file).logger


def normalize_column(col):
    """Min-Max归一化列"""
    min_val = col.min()
    max_val = col.max()
    return (col - min_val) / (max_val - min_val)


def load_and_preprocess_data(log):
    """
    加载并预处理数据

    Returns:
        tuple: (train_data, test_data, all_data, item_info, trn_last_df, hot_item)
    """
    # 读取数据
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    item_raw_emb = pd.read_csv(ITEM_EMB_PATH)
    item_info = pd.read_csv(ITEM_INFO_PATH)

    # 预处理物品信息
    item_info = item_info.rename(columns={'article_id': 'click_article_id'})
    item_info['words_count'] = normalize_column(item_info['words_count'])
    item_info['created_at_ts'] = normalize_column(item_info['created_at_ts'])

    # 合并全量数据
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data = all_data.merge(item_info, how='left', on='click_article_id')
    all_data = all_data.sort_values(by=['user_id', 'click_timestamp']).reset_index(drop=True)

    # 获取热门物品
    hot_item = all_data['click_article_id'].value_counts().index[:350]

    # 提取用户最后一次点击作为验证集
    df_filter = all_data.groupby('user_id').filter(lambda x: len(x) > 1).reset_index(drop=True)
    trn_last_df = df_filter.groupby('user_id').tail(1).reset_index(drop=True)

    log.info(f"数据加载完成: 训练集{len(train_data)}条, 测试集{len(test_data)}条, 热门物品{len(hot_item)}个")

    return train_data, test_data, all_data, item_info, trn_last_df, hot_item


def extract_user_features(all_data, log):
    """
    提取用户特征（环境、设备等众数）

    Args:
        all_data: 全量数据
        log: 日志记录器

    Returns:
        pd.DataFrame: 添加了用户特征的数据
    """
    features = ['click_environment', 'click_deviceGroup', 'click_os',
                'click_country', 'click_region', 'click_referrer_type']

    for feature in features:
        mode_data = all_data.groupby('user_id')[feature].apply(lambda x: x.mode()[0]).reset_index()
        mode_data.rename(columns={feature: f'{feature}_mode'}, inplace=True)
        all_data = pd.merge(all_data, mode_data, on='user_id', how='left')

    log.info(f'用户特征提取完成:\n{all_data.head()}')
    return all_data


def encode_features(all_data, sparse_features, log):
    """
    对离散特征进行LabelEncoding

    Args:
        all_data: 全量数据
        sparse_features: 需要编码的稀疏特征列表
        log: 日志记录器

    Returns:
        tuple: (encoded_data, feature_max_idx, user_map, item_map)
    """
    feature_max_idx = {}
    user_map = {}
    item_map = {}

    log.info(f'编码前数据:\n{all_data[sparse_features].head()}')

    for feature in sparse_features:
        lbe = LabelEncoder()
        all_data[feature] = lbe.fit_transform(all_data[feature]) + 1
        feature_max_idx[feature] = all_data[feature].max() + 1

        # 建立编码映射
        if feature == 'user_id':
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
        if feature == 'click_article_id':
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    log.info(f'编码后数据:\n{all_data[sparse_features].head()}')
    return all_data, feature_max_idx, user_map, item_map


def build_dssm_model(feature_max_idx, user_cols, item_sparse, item_dense, seq_max_len, train_model_input, log):
    """
    构建DSSM模型

    Args:
        feature_max_idx: 特征最大编码值字典
        user_cols: 用户特征列
        item_sparse: 物品稀疏特征
        item_dense: 物品稠密特征
        seq_max_len: 序列最大长度
        train_model_input: 训练输入（用于构建负采样器）
        log: 日志记录器

    Returns:
        Model: 编译好的DSSM模型
    """
    # 定义用户特征
    user_features = [SparseFeat('user_id', feature_max_idx['user_id'], 32)]
    user_features += [SparseFeat(feat, feature_max_idx[feat], 16) for feat in user_cols[1:]]
    user_features += [
        VarLenSparseFeat(
            SparseFeat("hist_click_article_id", feature_max_idx["click_article_id"], 32, embedding_name="click_article_id"),
            seq_max_len, 'sum', 'hist_len', 'hist_weight', False
        ),
        VarLenSparseFeat(
            SparseFeat("hist_category_id", feature_max_idx["category_id"], 32, embedding_name="category_id"),
            seq_max_len, 'sum', 'hist_len', 'hist_weight', False
        ),
    ]

    # 定义物品特征
    item_features = [SparseFeat(feat, feature_max_idx[feat], 32) for feat in item_sparse]
    item_features += [DenseFeat(feat) for feat in item_dense]

    # 模型配置
    dnn_layer = (64, 32)
    BN = False
    optimizer = 'adam'

    # 构建负采样配置（inbatch负采样）
    sample_method = 'inbatch'
    train_counter = Counter(train_model_input['click_article_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_features[0].vocabulary_size)]
    sampler_config = NegativeSampler(sample_method, num_sampled=0, item_name="click_article_id", item_count=item_count)

    # 构建模型
    model = DSSM(
        user_features, item_features,
        user_dnn_hidden_units=dnn_layer,
        item_dnn_hidden_units=dnn_layer,
        loss_type='softmax',
        dnn_use_bn=BN,
        sampler_config=sampler_config
    )
    model.compile(optimizer=optimizer, loss=sampledsoftmaxloss)

    log.info(f'模型配置: dnn={dnn_layer}, optimizer={optimizer}, loss=sampledsoftmaxloss, sampler={sample_method}')

    return model


def train_dssm_model(model, train_model_input, train_label, log):
    """
    训练DSSM模型

    Args:
        model: DSSM模型
        train_model_input: 训练输入
        train_label: 训练标签
        log: 日志记录器

    Returns:
        History: 训练历史
    """
    bs = 256
    epoch = 2
    vr = 0.0

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    history = model.fit(train_model_input, train_label, batch_size=bs, epochs=epoch, verbose=1, validation_split=vr)

    log.info(f'训练配置: bs={bs}, epoch={epoch}, vr={vr}')

    # 保存模型权重
    model.save_weights(MODEL_PATH / 'dssm_weights.h5')
    log.info('模型权重已保存')

    return history


def generate_embeddings(model, test_model_input, item_profile, user_map, item_map):
    """
    生成用户和物品的Embedding向量

    Args:
        model: 训练好的DSSM模型
        test_model_input: 测试输入
        item_profile: 物品画像
        user_map: 用户ID映射
        item_map: 物品ID映射

    Returns:
        tuple: (user_embs, item_embs, index_to_itemid)
    """
    # 构建物品输入
    all_item_model_input = {
        "click_article_id": item_profile['click_article_id'].values,
        "category_id": item_profile['category_id'].values,
        "words_count": item_profile['words_count'].values,
        "created_at_ts": item_profile['created_at_ts'].values
    }

    # 获取Embedding模型
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # 预测Embedding
    user_embs = user_embedding_model.predict(test_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # 保存Embedding
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    raw_user_id_emb_dict = {user_map[k]: v for k, v in zip(test_model_input['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_map[k]: v for k, v in zip(item_profile['click_article_id'], item_embs)}

    pickle.dump(raw_user_id_emb_dict, open(SAVE_DIR / 'user_embedding_dssm_part.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(SAVE_DIR / 'item_embedding_dssm_part.pkl', 'wb'))

    # 构建索引映射
    index_to_itemid = {idx: item_profile['click_article_id'].iloc[idx] for idx in range(len(item_embs))}

    return user_embs, item_embs, index_to_itemid


def process_chunk(task):
    """
    处理任务分片，进行向量召回

    Args:
        task: 包含(chunk, trn_user_item_time, user_map, index_to_itemid, item_map, trn_last_df, hot_item, recall_item_num)

    Returns:
        list: 召回结果DataFrame列表
    """
    (chunk, trn_user_item_time, user_map, index_to_itemid, item_map, trn_last_df, hot_item, recall_item_num) = task
    chunk_results = []

    for target_idx, sim_value_list, rele_idx_list in chunk:
        # 将 numpy int 转换为 Python int（确保类型一致）
        target_idx = int(target_idx)

        # 获取该用户的历史点击记录
        # 注意：测试集中可能有训练集中没有的新用户，这些用户可能不在 trn_user_item_time 中
        if target_idx not in trn_user_item_time:
            # 新用户没有历史记录，使用空列表
            user_hist_items = []
        else:
            user_hist_items = trn_user_item_time[target_idx]

        user_hist_items_set = {item_id for item_id, _ in user_hist_items}
        target_raw_id = user_map[target_idx]  # 编码ID -> 原始ID

        # 构造召回字典，累计相似度分数
        recall_dict = {}
        for rele_idx, sim_value in zip(rele_idx_list, sim_value_list):
            if sim_value < 0.05:
                continue
            rele_idx = index_to_itemid[rele_idx]
            if rele_idx in user_hist_items_set or rele_idx == 0:
                continue
            rele_raw_id = item_map[rele_idx]
            recall_dict[rele_raw_id] = recall_dict.get(rele_raw_id, 0) + sim_value

        # 按相似度排序并取前100
        sim_items = sorted(recall_dict.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 构造结果DataFrame
        df_temp = pd.DataFrame({
            'article_id': item_ids,
            'sim_score': item_sim_scores,
            'user_id': target_raw_id
        })

        # 打标签
        if trn_last_df is not None:
            last_item_id = trn_last_df[trn_last_df['user_id'] == target_raw_id]['click_article_id'].values[0]
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == last_item_id, 'label'] = 1
        else:
            df_temp['label'] = -1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        chunk_results.append(df_temp)

    return chunk_results


def faiss_recall(user_embs, item_embs, test_model_input, trn_user_item_time, user_map, index_to_itemid, item_map, trn_last_df, hot_item, log):
    """
    使用Faiss进行向量召回

    Args:
        user_embs: 用户Embedding
        item_embs: 物品Embedding
        test_model_input: 测试输入（包含user_id用于映射）
        trn_user_item_time: 用户历史点击序列
        user_map: 用户ID映射
        index_to_itemid: 索引到物品ID映射
        item_map: 物品ID映射
        trn_last_df: 验证集标签
        hot_item: 热门物品
        log: 日志记录器

    Returns:
        pd.DataFrame: 召回结果
    """
    # 使用Faiss进行最近邻搜索
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    sim, idx = index.search(np.ascontiguousarray(user_embs), 200)

    log.info(f'相似度范围: [{np.min(sim):.4f}, {np.max(sim):.4f}]')

    # 多进程并行处理召回结果
    # 注意：必须使用 test_model_input['user_id'] 而不是 range(len(user_embs))
    # 因为 user_embs 的顺序与 test_model_input['user_id'] 对应，需要用正确的用户ID查找历史记录
    data_inputs = list(zip(test_model_input['user_id'], sim, idx))
    cpu_nums = mp.cpu_count()
    chunk_size = int(math.ceil(len(data_inputs) / cpu_nums))
    chunks = [(data_inputs[i * chunk_size: (i + 1) * chunk_size],
               trn_user_item_time, user_map, index_to_itemid, item_map, trn_last_df, hot_item, 100)
              for i in range(cpu_nums)]

    with mp.Pool(processes=cpu_nums) as pool:
        results = list(pool.imap_unordered(process_chunk, chunks))

    data_list = [df for sublist in results for df in sublist]
    df_data = pd.concat(data_list, ignore_index=True)

    return df_data


def evaluate_recall(df_data, trn_last_df, log):
    """
    评估召回效果

    Args:
        df_data: 召回结果
        trn_last_df: 验证集标签
        log: 日志记录器
    """
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    df_data_train = df_data[df_data['user_id'] < 200000]
    total = trn_last_df[trn_last_df['user_id'] < 200000]['user_id'].nunique()

    metrics = evaluate(df_data_train[df_data_train['label'].notnull()], total, 'article_id')
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = metrics

    log.info(f'\n===== 验证集评估结果 =====')
    log.info(f'HitRate@5:  {hitrate_5:.4f}  |  MRR@5:  {mrr_5:.4f}')
    log.info(f'HitRate@10: {hitrate_10:.4f}  |  MRR@10: {mrr_10:.4f}')
    log.info(f'HitRate@20: {hitrate_20:.4f}  |  MRR@20: {mrr_20:.4f}')
    log.info(f'HitRate@40: {hitrate_40:.4f}  |  MRR@40: {mrr_40:.4f}')
    log.info(f'HitRate@50: {hitrate_50:.4f}  |  MRR@50: {mrr_50:.4f}')


def generate_test_submit(model, all_data, user_profile, user_map, item_map, index_to_itemid, item_embs, trn_user_item_time, hot_item, log):
    """
    生成测试集提交结果

    Args:
        model: 训练好的模型
        all_data: 全量数据
        user_profile: 用户画像
        user_map: 用户ID映射
        item_map: 物品ID映射
        index_to_itemid: 索引到物品ID映射
        item_embs: 物品Embedding
        trn_user_item_time: 用户历史点击序列
        hot_item: 热门物品
        log: 日志记录器
    """
    seq_max_len = 30
    train_cols = ['user_id', 'hist_click_article_id', 'click_article_id', 'label', 'hist_len', 'hist_cates', 'hist_weight']

    # 构建推理数据
    # 注意：user_cols 必须包含 'user_id'，因为模型需要它作为输入
    infer_model_input = build_infer_data(all_data, user_profile, user_profile.columns.tolist(), seq_max_len, train_cols)

    # 获取用户Embedding
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    user_embs_infer = user_embedding_model.predict(infer_model_input, batch_size=2 ** 12)

    # Faiss搜索
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    sim, idx = index.search(np.ascontiguousarray(user_embs_infer), 200)

    log.info(f'测试集相似度范围: [{np.min(sim):.4f}, {np.max(sim):.4f}]')

    # 多进程召回
    data_inputs = list(zip(infer_model_input['user_id'], sim, idx))
    cpu_nums = mp.cpu_count()
    chunk_size = int(math.ceil(len(data_inputs) / cpu_nums))
    chunks = [(data_inputs[i * chunk_size: (i + 1) * chunk_size],
               trn_user_item_time, user_map, index_to_itemid, item_map, None, hot_item, 100)
              for i in range(cpu_nums)]

    with mp.Pool(processes=cpu_nums) as pool:
        results = list(pool.imap_unordered(process_chunk, chunks))

    data_list = [df for sublist in results for df in sublist]
    df_data = pd.concat(data_list, ignore_index=True)

    # 只保留测试集用户 (user_id >= 200000)
    df_data = df_data[df_data['user_id'] >= 200000]

    # 保存结果
    RECALL_DICT_PATH.mkdir(parents=True, exist_ok=True)
    df_data.to_pickle(RECALL_DICT_PATH / 'dssm_tst_recall.pkl')

    # 生成提交文件
    gen_infer_submit(df_data, SUBMIT_PATH, 'dssm')
    log.info('测试集提交文件已生成')


def main():
    """主函数"""
    # ==================== 1. 初始化 ====================
    log = setup_logger()
    log.info("=" * 60)
    log.info("DSSM 召回模型启动")
    log.info("=" * 60)

    # ==================== 2. 数据加载与预处理 ====================
    log.info("【步骤1】加载并预处理数据...")
    _, _, all_data, item_info, trn_last_df, hot_item = load_and_preprocess_data(log)

    # 调试模式：只使用少量数据快速验证代码逻辑
    if DEBUG_MODE:
        log.info("=" * 60)
        log.info("【调试模式】只使用100个用户的数据进行测试")
        log.info("=" * 60)
        sample_users = all_data['user_id'].unique()[:100]
        all_data = all_data[all_data['user_id'].isin(sample_users)].reset_index(drop=True)
        trn_last_df = trn_last_df[trn_last_df['user_id'].isin(sample_users)].reset_index(drop=True)
        log.info(f"采样后数据量: {len(all_data)} 条记录, {len(sample_users)} 个用户")

    # ==================== 3. 特征工程 ====================
    log.info("【步骤2】特征工程...")
    all_data = extract_user_features(all_data, log)

    # 定义特征列
    user_cols = ["user_id", 'click_environment_mode', 'click_deviceGroup_mode',
                 'click_os_mode', 'click_country_mode', 'click_region_mode', 'click_referrer_type_mode']
    item_cols = ['click_article_id', "category_id", "words_count", "created_at_ts"]
    sparse_features = ['user_id', 'click_article_id', 'category_id', 'click_environment_mode',
                       'click_deviceGroup_mode', 'click_os_mode', 'click_country_mode',
                       'click_region_mode', 'click_referrer_type_mode']

    # ==================== 4. 特征编码 ====================
    log.info("【步骤3】特征编码...")
    all_data, feature_max_idx, user_map, item_map = encode_features(all_data, sparse_features, log)

    # 提取用户/物品画像
    user_profile = all_data[user_cols].drop_duplicates('user_id').reset_index(drop=True)
    item_profile = all_data[item_cols].drop_duplicates('click_article_id').reset_index(drop=True)

    # ==================== 5. 构建训练数据 ====================
    log.info("【步骤4】构建训练数据...")
    seq_max_len = 30
    train_cols = ['user_id', 'hist_click_article_id', 'click_article_id', 'label', 'hist_len', 'hist_cates', 'hist_weight']

    # 根据配置选择训练数据
    # True:  使用全量数据训练（用于最终提交，无验证标签）
    # False: 使用去掉最后一次点击的数据训练（留出最后一次用于离线验证）
    if USE_FULL_DATA_FOR_TRAINING:
        log.info("使用全量数据训练（用于最终提交）")
        train_df = all_data
        trn_hist_df = all_data  # 全量数据没有trn_last_df的概念
    else:
        log.info("使用去掉最后一次点击的数据训练（留出验证集）")
        # 提取历史序列（除最后一次）
        df_filter = all_data.groupby('user_id').filter(lambda x: len(x) > 1)
        df_filter = df_filter.sort_values(by=['user_id', 'click_timestamp'], ascending=True).reset_index(drop=True)
        trn_hist_df = df_filter.groupby('user_id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        train_df = trn_hist_df

    train_model_input, train_label, test_model_input = build_train_test_data(
        train_df, user_profile, item_profile, user_cols, item_cols, train_cols, seq_max_len
    )

    # ==================== 6. 构建和训练模型 ====================
    log.info("【步骤5】构建DSSM模型...")
    item_sparse = ['click_article_id', "category_id"]
    item_dense = ["words_count", "created_at_ts"]

    model = build_dssm_model(feature_max_idx, user_cols, item_sparse, item_dense, seq_max_len, train_model_input, log)

    log.info("【步骤6】训练模型...")
    train_dssm_model(model, train_model_input, train_label, log)

    # ==================== 7. 生成Embedding并召回 ====================
    log.info("【步骤7】生成Embedding向量...")
    user_embs, item_embs, index_to_itemid = generate_embeddings(
        model, test_model_input, item_profile, user_map, item_map
    )

    log.info(f"用户Embedding形状: {user_embs.shape}, 物品Embedding形状: {item_embs.shape}")

    # ==================== 8. 验证集召回与评估（仅当使用非全量数据时） ====================
    if not USE_FULL_DATA_FOR_TRAINING:
        log.info("\n" + "=" * 60)
        log.info("【阶段1】验证集召回与评估")
        log.info("=" * 60)

        trn_user_item_time = gen_user_item_time(trn_hist_df)
        df_recall = faiss_recall(user_embs, item_embs, test_model_input, trn_user_item_time, user_map, index_to_itemid,
                                  item_map, trn_last_df, hot_item, log)

        # 保存召回结果
        RECALL_DICT_PATH.mkdir(parents=True, exist_ok=True)
        df_recall.to_pickle(RECALL_DICT_PATH / 'dssm_recall.pkl')

        # 评估
        evaluate_recall(df_recall, trn_last_df, log)
    else:
        log.info("使用全量数据训练，跳过验证集评估阶段")

    # ==================== 9. 测试集预测 ====================
    log.info("\n" + "=" * 60)
    log.info("【阶段2】测试集预测")
    log.info("=" * 60)

    all_user_item_time = gen_user_item_time(all_data)
    generate_test_submit(model, all_data, user_profile, user_map, item_map,
                         index_to_itemid, item_embs, all_user_item_time, hot_item, log)

    log.info("\n" + "=" * 60)
    log.info("DSSM 召回模型运行完成")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
