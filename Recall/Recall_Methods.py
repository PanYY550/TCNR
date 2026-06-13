import sys
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import warnings, math, pickle
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import faiss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.utils import gen_user_item_time

warnings.filterwarnings('ignore')

#########################################itemcf构建相似矩阵###########################################
def itemcf_sim_process_chunk(args):
    """
    处理单个用户的行为序列，计算部分共现矩阵。
    
    参数:
    args (tuple): 包含以下内容：
        - user_click_seq: 用户的行为序列，格式为 [(item_id, click_timestamp)]
        - item_creat_ts_norm_dict: 物品创建时间归一化字典
        - strategy: 计算策略
        
        补充：
        user_click_seq_chunk :一批用户的点击序列（如125个用户）
        item_creat_ts_norm_dict :物品创建时间归一化字典 {item_id: norm_time}

    返回:
    tuple: (item_cnt, item_sim_dict)，分别是部分物品计数器和部分相似度字典。
            item_cnt :物品被点击次数计数器 {item_id: cnt}
            item_sim_dict :物品相似度字典 {item_id: {item_id: sim}}
            eg：
            item_cnt = {A: 10, B: 5, C: 8}           # 各物品被点击次数
            item_sim_dict = {A: {B: 2.5, C: 1.8},         # A与B、C的相似度分数 
                             B: {A: 2.5, C: 0.9},
                             C: {A: 1.8, B: 0.9}}
    """
    user_click_seq_chunk,item_creat_ts_norm_dict = args   # 从参数中提取用户点击序列和物品创建时间归一化字典
    item_cnt = defaultdict(int)   # 物品被点击次数计数器 首次访问时自动初始化为 0 不用检查 key 是否存在，直接 += 1 即可
    item_sim_dict = {}   # 物品相似度字典 {item_id: {item_id: sim}}

    for user_click_seq in tqdm(user_click_seq_chunk):


        for loc1, (item_idi, click_tsi) in enumerate(user_click_seq):  
            item_cnt[item_idi] += 1    # 统计物品被点击次数
            item_sim_dict.setdefault(item_idi, {}) # 初始化该物品的相似度字典
 

            for loc2, (item_idj, click_tsj) in enumerate(user_click_seq):
                if item_idj == item_idi:
                    continue
    
                # 计算权重
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # created_time_weight = np.exp(0.8 ** np.abs(item_creat_ts_norm_dict[item_idi] - item_creat_ts_norm_dict[item_idj]))

                # 根据位置权重和用户活跃度（点击总长度）更新相似度
                item_sim_dict[item_idi].setdefault(item_idj, 0)
                item_sim_dict[item_idi][item_idj] += loc_weight / math.log(len(user_click_seq) + 1)

    return item_cnt, item_sim_dict

def itemcf_sim_cal_parallel(click_sequence_dict,item_creat_ts_norm_dict,  sim_m_path, log=None):
    '''
    click_sequence_dict(dict): user:[(item,time),(item,time)] 按时间升序
    item_creat_ts_norm_dict(dict): item:created_at_ts 物品创建时间归一化 {item_id: norm_time}
    sim_m_path(str): 相似矩阵保存路径
    log: 日志对象
    '''
    if log:
        log.info('遍历各用户行为序列，计算共现矩阵...')
    else:
        print('遍历各用户行为序列，计算共现矩阵...')
    user_click_seq_list = list(click_sequence_dict.values()) # 取出所有用户的点击序列
    num_users = len(user_click_seq_list) # 用户总数
    num_cpus = cpu_count() # cpu核心数
    # 计算每个核心需要处理的任务量
    chunk_size = (num_users + num_cpus - 1) // num_cpus  # 向上取整
    chunks = [user_click_seq_list[i:i + chunk_size] for i in range(0, num_users, chunk_size)]

    # 准备多进程任务
    tasks = [(chunk,item_creat_ts_norm_dict) for chunk in chunks]

    # 使用多进程池并行计算
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap_unordered(itemcf_sim_process_chunk, tasks))

    # 进程一:
    # item_cnt = {A: 10, B: 5, C: 8}           # 各物品被点击次数
    # item_sim = {A: {B: 2.5, C: 1.8},         # A与B、C的相似度分数
    #             B: {A: 2.5, C: 0.9},
    #             C: {A: 1.8, B: 0.9}}

    # 进程2的结果：
    # item_cnt = {A: 5, B: 12, D: 3}           # 注意有物品D
    # item_sim = {A: {B: 1.2, D: 0.5},
    #             B: {A: 1.2, D: 0.3},
    #             D: {A: 0.5, B: 0.3}}

    # 合并结果
    # item_cnt = {A: 15, B: 17, C: 8, D: 3}   # 10+5, 5+12, 8+0, 0+3
    # item_sim_dict = {
    #     A: {B: 3.7, C: 1.8, D: 0.5},      # B: 2.5+1.2, C: 1.8+0, D: 0+0.5
    #     B: {A: 3.7, C: 0.9, D: 0.3},
    #     C: {A: 1.8, B: 0.9},
    #     D: {A: 0.5, B: 0.3}
    # }
    # 合并结果
    item_cnt = defaultdict(int)
    item_sim_dict = {}
    for partial_item_cnt, partial_item_sim_dict in results:
        for item_id, cnt in partial_item_cnt.items():
            item_cnt[item_id] += cnt
        for item_idi, sim_items in partial_item_sim_dict.items():
            item_sim_dict.setdefault(item_idi, {})
            for item_idj, sim_score in sim_items.items():
                item_sim_dict[item_idi].setdefault(item_idj, 0)
                item_sim_dict[item_idi][item_idj] += sim_score

    # 归一化处理
    for item_idi, sim_items in item_sim_dict.items():
        for item_idj, sim_score in sim_items.items():
            item_sim_dict[item_idi][item_idj] = sim_score / np.sqrt(item_cnt[item_idi] * item_cnt[item_idj])

    # 保存结果
    pickle.dump(item_sim_dict, open(sim_m_path, 'wb'))
    if log:
        log.info(f'itemcf sim matrix build successfully!')

def itemcf_sim_cal(click_sequence_dict,item_creat_ts_norm_dict,sim_m_path,log=None):
    '''
    不并行计算物品共现矩阵
    click_sequence_dict(dict): user:[(item,time),(item,time)] 按时间升序
    '''
    if ~log:
        print('遍历各用户行为序列，计算共现矩阵...')
    else:
        log.info('遍历各用户行为序列，计算共现矩阵...')
    
    # 初始化物品计数器和相似度字典
    item_cnt = defaultdict(int)  # 记录每个物品的出现次数
    item_sim_dict = {}  # 记录物品之间的相似度

    # 遍历每个用户的行为序列
    for user_id, click_seq_list in tqdm(click_sequence_dict.items()):
        # 遍历用户行为序列中的每个点击事件
        for loc1, (item_idi, click_tsi) in enumerate(click_seq_list):
            # 初始化当前物品的计数和相似度字典
            item_cnt.setdefault(item_idi, 0)
            item_cnt[item_idi] += 1  # 更新物品的点击次数
            item_sim_dict.setdefault(item_idi, {})  # 初始化当前物品的相似度字典

            # 遍历用户行为序列中的其他点击事件，计算共现权重
            for loc2, (item_idj, click_tsj) in enumerate(click_seq_list):
                if item_idj == item_idi:
                    continue  # 如果是同一个物品，跳过

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7  # 正向点击权重为1.0，反向点击权重为0.7
                # 位置信息权重，参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 计算物品创建时间权重
                # created_time_weight = np.exp(0.8 ** np.abs(item_creat_ts_norm_dict[item_idi] - item_creat_ts_norm_dict[item_idj]))
                
                # 初始化当前物品对的相似度
                item_sim_dict[item_idi].setdefault(item_idj, 0)
                # 更新相似度得分，考虑点击时间权重和位置权重
                item_sim_dict[item_idi][item_idj] += loc_weight / math.log(len(click_seq_list) + 1)



    # 对相似度矩阵进行归一化处理
    for item_idi, sim_items in item_sim_dict.items():
        for item_idj, sim_score in sim_items.items():
            # 使用余弦相似度归一化公式
            item_sim_dict[item_idi][item_idj] = sim_score / np.sqrt(item_cnt[item_idi] * item_cnt[item_idj])

    # 将相似度字典保存到文件中
    pickle.dump(item_sim_dict, open(sim_m_path, 'wb'))
    if log:
        log.info(f'itemcf sim matrix build successfully!')



#######################################根据itemcf相似度矩阵召回物品##################################
def i2i_recall_process_chunk(args):
    """
    处理一个用户块的推荐逻辑。

    参数:
    args (tuple): 包含以下内容：
        - user_chunk: 用户块，包含多个用户的 ID
        - trn_user_item_time: 用户历史行为序列字典
        - i2i_matrix: 物品相似度矩阵
        - item_creat_ts_norm_dict: 物品创建时间归一化字典
        - hot_item: 热门文章列表
        - sim_item_topk: 每个物品的相似物品数量
        - recall_item_num: 每个用户的召回物品数量
        - last_df: 最后一次点击行为的 DataFrame（可选）

    返回:
    pd.DataFrame: 包含用户块推荐结果的 DataFrame。
    """
    user_chunk, trn_user_item_time, i2i_matrix,item_creat_ts_norm_dict, hot_item, sim_item_topk, recall_item_num, last_df = args
    data_list = []

    for user_id in tqdm(user_chunk):
        user_hist_items = trn_user_item_time[user_id]
        user_hist_items_ = {item_id for item_id, _ in user_hist_items}
        item_rank = {}

        # 遍历用户历史交互的文章及其点击时间
        for loc, (i, click_time) in enumerate(user_hist_items[-2:]):
            # 获取与当前文章最相似的前 sim_item_topk 篇文章
            for j, wij in sorted(i2i_matrix.get(i, {}).items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items_:
                    continue
                # 考虑文章的创建时间权重
                created_time_weight = np.exp(0.8 ** np.abs(item_creat_ts_norm_dict[i] - item_creat_ts_norm_dict[j]))
                # 计算位置权重
                loc_weight = (0.9 ** (len(user_hist_items[-2:]) - loc))
            
                # 更新文章 j 的推荐分数
                item_rank.setdefault(j, 0)
                item_rank[j] += wij * loc_weight*created_time_weight
        # 如果推荐的文章数量不足，用热门文章补全
        # if len(item_rank) < recall_item_num:
        #     for i, item in enumerate(hot_item):
        #         if item in item_rank:
        #             continue
        #         item_rank[item] = - i - 100
        #         if len(item_rank) == recall_item_num:
        #             break

        # 对推荐列表按分数排序
        sim_items = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

        # 构建 DataFrame
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        df_temp = pd.DataFrame({'article_id': item_ids, 'sim_score': item_sim_scores, 'user_id': user_id})

        # 打标签
        if last_df is not None:
            last_item_id = last_df[last_df['user_id'] == user_id]['click_article_id'].values[0]
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == last_item_id, 'label'] = 1
        else:
            df_temp['label'] = -1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    return pd.concat(data_list, ignore_index=True)

def I2I_recall_parallel(i2i_matrix, hist_df,item_creat_ts_norm_dict, hot_item, method, sim_item_topk, recall_item_num, recall_dict_path, last_df=None):
    """
    并行化实现基于物品协同过滤的召回逻辑。

    参数:
    i2i_matrix (dict): 物品相似度矩阵
    hist_df (pd.DataFrame): 用户历史行为数据
    item_creat_ts_norm_dict (dict): 物品创建时间归一化字典
    hot_item (list): 热门文章列表
    method (str): 召回方法名称
    sim_item_topk (int): 每个物品的相似物品数量
    recall_item_num (int): 每个用户的召回物品数量
    recall_dict_path (Path): 召回结果保存路径
    last_df (pd.DataFrame): 最后一次点击行为的 DataFrame（可选）
    """
    # 生成用户历史行为序列
    trn_user_item_time = gen_user_item_time(hist_df)

    # 将用户均匀分配到每个 CPU 核心
    user_ids = hist_df['user_id'].unique()
    num_users = len(user_ids)
    num_cpus = cpu_count()
    chunk_size = (num_users + num_cpus - 1) // num_cpus  # 每个核心处理的任务量
    user_chunks = [user_ids[i:i + chunk_size] for i in range(0, num_users, chunk_size)]

    # 准备多进程任务
    tasks = [
        (user_chunk, trn_user_item_time, i2i_matrix,item_creat_ts_norm_dict, hot_item, sim_item_topk, recall_item_num, last_df)
        for user_chunk in user_chunks
    ]

    # 使用多进程池并行计算
    with Pool(processes=num_cpus) as pool:
        results = list(pool.imap_unordered(i2i_recall_process_chunk, tasks))

    # 合并结果
    df_data = pd.concat(results, ignore_index=True)

    # 保存结果
    os.makedirs(recall_dict_path, exist_ok=True)
    df_data.to_pickle(recall_dict_path / f'{method}_recall.pkl')

def I2I_recall(i2i_matrix,hist_df,hot_item,method,sim_item_topk,recall_item_num,recall_dict_path,last_df=None):
    trn_user_item_time = gen_user_item_time(hist_df)
    data_list=[]
    for user_id in tqdm(hist_df['user_id'].unique()):
        user_hist_items = trn_user_item_time[user_id]
        # 将用户历史交互的文章转换为集合，方便后续快速查找
        user_hist_items_ = {item_id for item_id, _ in user_hist_items}

        # 初始化一个空字典，用于存储文章及其对应的推荐分数
        item_rank = {}
        # 遍历用户历史交互的文章及其点击时间，loc表示文章在历史序列中的位置
        for loc, (i, click_time) in enumerate(user_hist_items[-2:]):
            # 对当前文章i，获取与其最相似的前sim_item_topk篇文章，并遍历这些相似文章
            for j, wij in sorted(i2i_matrix.get(i, {}).items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                # 如果相似文章j已经在用户的历史交互文章中，跳过
                if j in user_hist_items_:
                    continue
                
                # 计算文章在用户历史点击序列中的位置权重，位置越靠后，权重越小
                loc_weight = (0.9 ** (len(user_hist_items[-2:]) - loc))
                    
                # 将文章j的推荐分数初始化为0（如果尚未初始化）
                item_rank.setdefault(j, 0)
                # 更新文章j的推荐分数，考虑位置权重
                item_rank[j] += wij * loc_weight

        # 如果推荐的文章数量不足recall_item_num个，用热门文章补全
        if len(item_rank) < recall_item_num:
            # 遍历点击次数最多的文章列表
            for i, item in enumerate(hot_item):
                # 如果文章已经在推荐列表中，跳过
                if item in item_rank.items():
                    continue
                # 给热门文章一个负数的分数，确保它们排在推荐列表的末尾
                item_rank[item] = - i - 100
                # 如果推荐列表已经达到recall_item_num个，停止补全
                if len(item_rank) == recall_item_num:
                    break

        # 对推荐列表按分数从高到低排序，并取前recall_item_num个文章
        sim_items = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

        # 根据最后一次点击行为打标签
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id
        if last_df:
            last_item_id = last_df[last_df['user_id']==user_id]['click_article_id'].values[0]
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == last_item_id, 'label'] = 1
        else:
            df_temp['label'] = -1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs(recall_dict_path, exist_ok=True)
    df_data.to_pickle(recall_dict_path / f'{method}_recall.pkl')

# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item

########################################基于语义embedding#########################################

def embdding_sim(item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :param topk: 找最相似的topk篇
        return 文章相似性矩阵
        
        思路: 对于每一篇文章，基于embedding的相似性返回topk个与其最相似的文章，只不过由于文章数量太多，这里用了faiss进行加速
    """
    if isinstance(item_emb_df, pd.DataFrame):
        item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
        item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x] 
        # 将embedding列的值转换为NumPy数组，并确保数组是连续的（ascontiguousarray）
        # 数据类型为float32，因为faiss对float32的支持更好
        item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    elif isinstance(item_emb_df,dict):
        raw_ids = list(item_emb_df.keys())
        embeddings = [np.array(item_emb_df[raw_id]) for raw_id in raw_ids]
        item_emb_np = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
        item_idx_2_rawid_dict = { idx: raw_id for idx, raw_id in enumerate(raw_ids) }

    
    # 建立faiss索引
    # 使用faiss的IndexFlatIP（内积索引），因为向量已经归一化，内积等价于余弦相似度
    # item_emb_np.shape[1]表示embedding向量的维度
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    
    # 将归一化后的embedding向量添加到faiss索引中
    item_index.add(item_emb_np)
    
    # 相似度查询
    # 对每个embedding向量，使用faiss搜索其topk个最相似的向量
    # sim: 相似度矩阵，shape为 (n_items, topk)，存储每个item与topk个item的相似度
    # idx: 索引矩阵，shape为 (n_items, topk)，存储每个item对应的topk个item的索引
    sim, idx = item_index.search(item_emb_np, topk)
    print(np.min(np.min(sim)))
    
    # 将向量检索的结果保存成原始id的对应关系
    # 使用defaultdict存储相似度字典，格式为 {target_item_id: {related_item_id: similarity_score}}
    item_sim_dict = defaultdict(dict)
    
    # 遍历每个item的检索结果
    # target_idx: 当前item的索引
    # sim_value_list: 当前item与topk个item的相似度列表
    # rele_idx_list: 当前item对应的topk个item的索引列表
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        # 获取当前item的原始id
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        
        # 从1开始遍历，跳过第一个（因为第一个是自己，相似度为1）
        # rele_idx: 相关item的索引
        # sim_value: 当前item与相关item的相似度
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            # 获取相关item的原始id
            # if(sim_value<0.2):
            #     continue
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            
            # 将相似度存储到字典中
            # 如果target_raw_id或rele_raw_id已经存在，则累加相似度（如果有重复）
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    
    # 保存i2i相似度矩阵
    # 使用pickle将item_sim_dict保存到指定路径
    pickle.dump(item_sim_dict, open(save_path, 'wb'))   
    
    # 返回相似度字典
    return item_sim_dict

def emb_recall_process_chunk(args):
    """
    处理一个用户块的推荐逻辑，并加入冷启动筛选规则。

    参数:
    args (tuple): 包含以下内容：
        - user_chunk: 用户块，包含多个用户的 ID
        - trn_user_item_time: 用户历史行为序列字典
        - i2i_matrix: 物品相似度矩阵
        - item_creat_ts_norm_dict: 物品创建时间归一化字典
        - hot_item: 热门文章列表
        - sim_item_topk: 每个物品的相似物品数量
        - recall_item_num: 每个用户的召回物品数量
        - last_df: 最后一次点击行为的 DataFrame（可选）
        - user_hist_item_typs_dict: 用户点击的文章的主题映射
        - user_hist_item_words_dict: 用户点击的历史文章的字数映射
        - user_last_item_created_time_dict: 用户点击的历史文章创建时间映射
        - item_type_dict: 文章主题映射
        - item_words_dict: 文章字数映射
        - click_article_ids_set: 用户点击过的文章集合

    返回:
    pd.DataFrame: 包含用户块推荐结果的 DataFrame。
    """
    user_chunk, trn_user_item_time, i2i_matrix, item_creat_ts_norm_dict, hot_item, sim_item_topk, recall_item_num,last_df = args
    data_list = []

    for user_id in tqdm(user_chunk):  
        user_hist_items = trn_user_item_time[user_id]  ## [(item_id, ts), ...]
        # user_hist_items = [(A, ts1), (B, ts2), (C, ts3)]
        # user_hist_items_ = {A, B, C}  # 集合
        user_hist_items_ = {item_id for item_id, _ in user_hist_items}  ## 转为集合，方便查重
        item_rank = {}   ## 存储候选物品分数

        # 遍历用户历史交互的文章及其点击时间
        for loc, (i, click_time) in enumerate(user_hist_items[-2:]):
            # 优化2：获取与当前文章最相似的前 sim_item_topk 篇文章    排序取TopN
            for j, wij in sorted(i2i_matrix[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items_:
                    continue

                # 计算文章创建时间差权重
                created_time_weight = np.exp(0.8 ** np.abs(item_creat_ts_norm_dict[i] - item_creat_ts_norm_dict[j]))
                # 计算位置权重
                loc_weight = (0.9 ** (len(user_hist_items[-2:]) - loc))

                # 更新文章 j 的推荐分数
                item_rank.setdefault(j, 0)
                item_rank[j] += wij * created_time_weight * loc_weight

        # 如果推荐的文章数量不足，用热门文章补全
        if len(item_rank) < recall_item_num:
            for i, item in enumerate(hot_item):
                if item in item_rank:
                    continue
                item_rank[item] = - i - 100
                if len(item_rank) == recall_item_num:
                    break

        # 对推荐列表按分数排序
        sim_items = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

        # 构建 DataFrame
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        df_temp = pd.DataFrame({'article_id': item_ids, 'sim_score': item_sim_scores, 'user_id': user_id})

        # 打标签
        if last_df is not None:
            last_item_id = last_df[last_df['user_id'] == user_id]['click_article_id'].values[0]
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == last_item_id, 'label'] = 1
        else:
            df_temp['label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    return pd.concat(data_list, ignore_index=True)

def emb_recall_parallel(i2i_matrix, hist_df, item_creat_ts_norm_dict, hot_item, method, sim_item_topk, recall_item_num,
                         recall_dict_path,last_df=None):
    """
    并行化实现基于物品协同过滤的召回逻辑，并加入冷启动筛选规则。

    参数:
    i2i_matrix (dict): 物品相似度矩阵
    hist_df (pd.DataFrame): 用户历史行为数据
    item_creat_ts_norm_dict (dict): 物品创建时间归一化字典
    hot_item (list): 热门文章列表
    method (str): 召回方法名称
    sim_item_topk (int): 每个物品的相似物品数量
    recall_item_num (int): 每个用户的召回物品数量
    recall_dict_path (Path): 召回结果保存路径
    last_df (pd.DataFrame): 最后一次点击行为的 DataFrame（可选）
    """
    # 生成用户历史行为序列  
    trn_user_item_time = gen_user_item_time(hist_df)

    # 将用户均匀分配到每个 CPU 核心
    user_ids = hist_df['user_id'].unique()
    num_users = len(user_ids)
    num_cpus = cpu_count()
    chunk_size = (num_users + num_cpus - 1) // num_cpus  # 每个核心处理的任务量
    user_chunks = [user_ids[i:i + chunk_size] for i in range(0, num_users, chunk_size)]

    # 准备多进程任务
    tasks = [
        (user_chunk, trn_user_item_time, i2i_matrix, item_creat_ts_norm_dict, hot_item, sim_item_topk, recall_item_num,last_df)
        for user_chunk in user_chunks
    ]

    # 使用多进程池并行计算
    with Pool(processes=num_cpus) as pool:
        results = list(pool.imap_unordered(emb_recall_process_chunk, tasks))

    # 合并结果
    df_data = pd.concat(results, ignore_index=True)

    # 保存结果
    os.makedirs(recall_dict_path, exist_ok=True)
    df_data.to_pickle(recall_dict_path / f'{method}_recall.pkl')

def evaluate(df, total,featname):
    hitrate_5 = 0
    mrr_5 = 0

    hitrate_10 = 0
    mrr_10 = 0

    hitrate_20 = 0
    mrr_20 = 0

    hitrate_40 = 0
    mrr_40 = 0

    hitrate_50 = 0
    mrr_50 = 0

    gg = df.groupby(['user_id'])

    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1][featname].values[0]
        except Exception as e:
            continue

        predictions = g[featname].values.tolist()

        rank = 0
        while predictions[rank] != item_id:
            rank += 1

        if rank < 5:
            mrr_5 += 1.0 / (rank + 1)
            hitrate_5 += 1

        if rank < 10:
            mrr_10 += 1.0 / (rank + 1)
            hitrate_10 += 1

        if rank < 20:
            mrr_20 += 1.0 / (rank + 1)
            hitrate_20 += 1

        if rank < 40:
            mrr_40 += 1.0 / (rank + 1)
            hitrate_40 += 1

        if rank < 50:
            mrr_50 += 1.0 / (rank + 1)
            hitrate_50 += 1

    hitrate_5 /= total
    mrr_5 /= total

    hitrate_10 /= total
    mrr_10 /= total

    hitrate_20 /= total
    mrr_20 /= total

    hitrate_40 /= total
    mrr_40 /= total

    hitrate_50 /= total
    mrr_50 /= total

    return hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50




