import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def build_train_test_data(trn_hist_df,user_profile,item_profile,user_cols,item_cols,train_cols,seq_max_len):

    trn_hist_df.sort_values("click_timestamp", inplace=True) # 按照点击时间排序
    
    # 初始化训练集和测试集
    train_set = []
    test_set = []


    # 对用户进行分组并遍历
    for reviewerID, hist in tqdm(trn_hist_df.groupby('user_id')):
        # 获取每个用户点击的文章列表（正样本）
        pos_list = hist['click_article_id'].tolist()
        cate_list = hist['category_id'].tolist()
        hist_weights = [0.7**(len(pos_list)-1-j) for j in range(len(pos_list))] # 按照点击的先后顺序，最近点击的文章权重为1，最远点击的文章权重为0.7^len(pos_list)
        l2_norm = np.linalg.norm(hist_weights)
        normalized_hist_weights = hist_weights / l2_norm if l2_norm != 0 else hist_weights
        normalized_hist_weights = normalized_hist_weights.tolist()  # 转换为列表
        
        if len(pos_list) == 1:
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, 1, [cate_list[0]],[1]))
            continue

        # 使用滑窗方法构造正样本
        for i in range(1, len(pos_list)):
            # 获取当前点击之前的历史文章列表
            hist_list = pos_list[:i]
            cate_list_hist = cate_list[:i]
            hist_w = np.array(hist_weights[:i])
            l2_norm = np.linalg.norm(hist_w)
            normalized_hist_w = hist_w / l2_norm if l2_norm != 0 else hist_w
            normalized_hist_w = normalized_hist_w.tolist()  # 转换为列表
            # 添加正样本到训练集
            train_set.append((reviewerID, hist_list[::-1], pos_list[i], 1, len(hist_list[::-1]), cate_list_hist[::-1],normalized_hist_w[::-1]))
        # 保留所有除最后一个点击文章的其他点击文章作为序列的样本得到测试集
        test_set.append((reviewerID, pos_list[::-1], pos_list[i], 1, len(pos_list[::-1]),cate_list[::-1],normalized_hist_weights[::-1]))
    # 打乱训练集和测试集的数据顺序，避免模型过拟合
    random.shuffle(train_set)
    train_df = pd.DataFrame(train_set, columns=train_cols)
    train_df = train_df.merge(user_profile,how='left',on='user_id')
    train_df = train_df.merge(item_profile,how='left',on='click_article_id')
    test_df = pd.DataFrame(test_set, columns=train_cols)
    test_df = test_df.merge(user_profile,how='left',on='user_id')
    train_seq = train_df['hist_click_article_id'].tolist()  # shape: (num_samples, variable_length)
    train_cate_seq = train_df['hist_cates'].tolist()
    train_cate_weight_seq = train_df['hist_weight'].tolist()
    test_seq = test_df['hist_click_article_id'].tolist()  # shape: (num_samples, variable_length)
    test_cate_seq =test_df['hist_cates'].tolist()
    test_cate_weight_seq = test_df['hist_weight'].tolist()

    ## 使用0填充至一样长度
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_cate_seq_pad = pad_sequences(train_cate_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0)
    train_cate_weight_seq = pad_sequences(train_cate_weight_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0,dtype='float32')

    test_seq_pad = pad_sequences(test_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    test_cate_seq_pad = pad_sequences(test_cate_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0)
    test_cate_weight_seq = pad_sequences(test_cate_weight_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0,dtype='float32')

    train_model_input = {}
    test_model_input = {}
    for col in user_cols + item_cols + ['hist_len']:
        train_model_input[col] = train_df[col].values
    train_model_input['hist_click_article_id'] = train_seq_pad  # numpy array
    train_model_input['hist_category_id'] = train_cate_seq_pad  # numpy array
    train_model_input['hist_weight'] = train_cate_weight_seq
    train_label = train_df['label'].values
    for col in user_cols + ['hist_len']:
        test_model_input[col] = test_df[col].values
    test_model_input['hist_click_article_id'] = test_seq_pad  # numpy array
    test_model_input['hist_category_id'] = test_cate_seq_pad  # numpy array
    test_model_input['hist_weight'] = test_cate_weight_seq

    return train_model_input,train_label,test_model_input

def build_infer_data(all_data,user_profile,user_cols,seq_max_len,cols):
    all_data.sort_values("click_timestamp", inplace=True)
    infer_set = []
    # 对用户进行分组并遍历
    for reviewerID, hist in tqdm(all_data.groupby('user_id')):
        # 获取每个用户点击的文章列表（正样本）
        pos_list = hist['click_article_id'].tolist()
        cate_list = hist['category_id'].tolist()
        hist_weights = [0.7**(len(pos_list)-1-j) for j in range(len(pos_list))]

        infer_set.append((reviewerID, pos_list[::-1], -1, -1, len(pos_list[::-1]),cate_list[::-1],hist_weights[::-1]))
 
    infer_df = pd.DataFrame(infer_set, columns=cols)
    infer_df = infer_df.merge(user_profile,how='left',on='user_id')
    infer_seq = infer_df['hist_click_article_id'].tolist()  # shape: (num_samples, variable_length)
    infer_cate_seq =infer_df['hist_cates'].tolist()
    infer_cate_weight_seq = infer_df['hist_weight'].tolist()

    infer_seq_pad = pad_sequences(infer_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    infer_cate_seq_pad = pad_sequences(infer_cate_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0)
    infer_cate_weight_seq = pad_sequences(infer_cate_weight_seq,maxlen=seq_max_len,padding='post',truncating='post',value=0,dtype='float32')

    infer_model_input = {}
    
    for col in user_cols + ['hist_len']:
        infer_model_input[col] = infer_df[col].values
    infer_model_input['hist_click_article_id'] = infer_seq_pad  # numpy array
    infer_model_input['hist_category_id'] = infer_cate_seq_pad  # numpy array
    infer_model_input['hist_weight'] = infer_cate_weight_seq

    return infer_model_input

def gen_infer_submit(df_data,save_dir,name):
    df_data['article_id'] = df_data['article_id'].astype(int)
    df_data = df_data[(df_data['user_id'] >= 200000) & (df_data['user_id'] <= 249999)]
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                ascending=[True,
                                            False]).reset_index(drop=True)
    df_top5 = df_data.groupby('user_id').head(5)

    # 第三步：将每个 user_id 的 top5 文章展开到同一行
    # 1) 重置索引，方便后续操作
    df_top5 = df_top5.reset_index(drop=True)
    # 2) 如果你想保留原先的排序信息（比如第 1 篇就对应 article_1 ...），
    #    可以给出一个 rank
    df_top5['rank'] = df_top5.groupby('user_id')['sim_score'].rank(method='first', ascending=False)

    # 3) pivot 将行转列
    df_pivot = df_top5.pivot(index='user_id', columns='rank', values='article_id')

    # 4) 由于 rank 的值是浮点数，这里先转成 int，然后根据它们是 1-5 这几个值重命名列
    df_pivot.columns = df_pivot.columns.astype(int)
    df_pivot = df_pivot.rename(columns={
        1: 'article_1',
        2: 'article_2',
        3: 'article_3',
        4: 'article_4',
        5: 'article_5'
    })

    # 5) 将 index 恢复成普通列
    df_pivot = df_pivot.reset_index()

    # 第四步：导出 CSV
    df_pivot.to_csv(save_dir / f'{name}_recom.csv', index=False, encoding='utf-8-sig')