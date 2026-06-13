import logging
import pandas as pd
import io

# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    
    return item_type_dict, item_words_dict, item_created_time_dict

def get_user_hist_item_info_dict(all_click):
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict
def print_df_info(df,log=None):
    if(log == None):
        print(df.info())
        print(pd.concat([df.head(),df.tail()],axis=0))
        print()
        print("nan data number: {}".format(df.isnull().sum().sum()))
    else:
        buffer = io.StringIO()
        df.info(buf=buffer)  # 将输出重定向到 buffer
        info_output = buffer.getvalue()
        log_message = (
        f"DataFrame Info:\n{info_output}\n\n"
        f"Head and Tail:\n{pd.concat([df.head(), df.tail()], axis=0)}\n\n"
        f"NaN Data Number: {df.isnull().sum().sum()}\n"
        )
        log.info(log_message)

def gen_user_item_time(df):
    """
    生成用户-物品-时间对的字典。

    参数:
    df (pd.DataFrame): 包含用户行为数据的 DataFrame，必须包含以下列：
        - user_id: 用户 ID
        - click_article_id: 点击的文章 ID
        - click_timestamp: 点击时间戳

    返回:
    dict: 一个字典，键是用户 ID，值是该用户按时间升序排列的 (文章 ID, 时间戳) 列表。
          例如：{user1: [(item1, time1), (item2, time2)], user2: [(item3, time3)]}
    """
    def make_pair(df):
        df = df.sort_values(by='click_timestamp',ascending=True)
        return list(zip(df['click_article_id'],df['click_timestamp']))
    df_group = df.groupby('user_id')[['click_article_id','click_timestamp']].apply(make_pair).reset_index().rename(columns={0: 'item_time_list'})
    return dict(zip(df_group['user_id'],df_group['item_time_list']))

class Logger(object):
    # 日志等级映射表，将自定义的字符串等级映射到 logging 内置的整数等级
    level_relations = {
        'debug': logging.DEBUG,    # 对应调试等级
        'info': logging.INFO,      # 对应一般信息
        'warning': logging.WARNING, # 对应警告信息
        'error': logging.ERROR,     # 对应错误信息
        'crit': logging.CRITICAL    # 对应严重错误
    }

    def __init__(
        self,
        filename,
        level='debug',
        fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: \n%(message)s'
    ):
        """
        初始化Logger对象。
        :param filename: 日志输出到的文件路径
        :param level: 日志等级（字符串，如 'debug'、'info' 等）
        :param fmt: 日志输出的格式（可使用 logging.Formatter 的格式占位符）
        """
        # 1. 使用 logging.getLogger 获取一个 logger 实例，名称可使用 filename 作为区分
        self.logger = logging.getLogger(filename)
        
        # 2. 创建一个日志格式化器，用来规定日志的最终输出格式
        #    这里的 fmt 中可包含 asctime, pathname, lineno, levelname, message 等常见字段
        format_str = logging.Formatter(fmt)
        
        # 3. 设置 logger 的日志等级（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        #    从 level_relations 字典中查找对应的等级数值
        self.logger.setLevel(self.level_relations.get(level))
        
        # 4. 创建一个“控制台”输出的 Handler（StreamHandler）
        #    用于把日志输出到终端/控制台
        sh = logging.StreamHandler()
        # 给控制台输出的 Handler 设置与 logger 相同的格式
        sh.setFormatter(format_str)
        
        # 5. 创建一个“文件”输出的 Handler（FileHandler）
        #    filename 参数指定日志写入文件；encoding 为 UTF-8；mode='a' 表示追加写入模式
        th = logging.FileHandler(filename=filename, encoding='utf-8', mode='a')
        # 同样给文件输出 Handler 设置格式
        th.setFormatter(format_str)
        
        # 6. 将这两个 Handler（控制台和文件）都添加到 logger 上
        #    这样日志既能输出到终端，又能写入文件
        self.logger.addHandler(sh)
        self.logger.addHandler(th)