import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from utils.augmentation import run_augmentation_single
from datasets import load_dataset
warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df.columns[1:]
            df_data = df[cols_data]
        elif self.features == "S":
            df_data = df[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df.columns[1:]
            df_data = df[cols_data]
        elif self.features == "S":
            df_data = df[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    '''
    由于原有的数据集处理逻辑是对时间序列进行整体切分，而我们所需要的处理逻辑是将现有的时间序列数据进行切分后拼接成一个个时间序列数据再拼接因此需要对这个数据集进行修改。
    '''
    def __init__(
        self,
        args,#无
        root_path,#无
        flag="train",
        size=None,
        total_seq_len=90,#add
        features="MS",
        data_path=None,
        target="sale_amount",
        scale=True,
        timeenc=1,
        train_only = False,#add
        freq="d",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.total_seq_len = total_seq_len
        # info
        if size == None:
            self.seq_len = 30
            self.label_len = 7
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.train_only = train_only#add
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        #更新数据集的使用逻辑
        if self.data_path is None:
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            df = dataset['train'].to_pandas()
        else:
            df = pd.read_parquet(self.data_path)
        #更新部分 数据预处理

        df = df.rename(columns={'dt':'date'})
        df = df.sort_values(by=['store_id','product_id','date'])
        df = df[
            ['date', 'discount', 'holiday_flag', 'precpt', 'avg_temperature', 'avg_humidity',
             'avg_wind_level', self.target]]
        df_stamp = df[['date']].copy()
        df_stamp['data'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.DatetimeIndex(df_stamp['data']),freq=self.freq).transpose()

        cols = list(df.columns)
        """
        df.columns: ['date', ...(other features), target feature]
        """
        cols = list(df.columns)
        cols.remove(self.target)
        cols.remove("date")
        df = df[["date"] + cols + [self.target]]#调整数据的列的前后顺序
        if self.features == "M" or self.features == "MS":
            cols_data = df.columns[1:]
            df_data = df[cols_data]
        elif self.features == "S":
            df_data = df[[self.target]]
        
        #列出时间序列开始和结束所对应的index

        num_train = int(self.total_seq_len * (0.7 if not self.train_only else 1))
        num_test = int(self.total_seq_len * 0.2)
        num_vali = self.total_seq_len - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.interval = border2 - border1

        df_stamp = df[["date"]].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1: #选择timeenc == 1
            data_stamp = time_features(
                pd.DatetimeIndex(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
       
        #需要先拼接后拟合
        if self.scale:
            # **核心缩放逻辑**:
            # 1. 从每个独立的时间序列（长度为 total_seq_len）中提取出训练部分。
            # 2. 将所有这些训练部分拼接在一起。
            # 3. 在这个拼接后的、纯粹的训练数据上拟合 StandardScaler。
            # 这样可以防止测试集和验证集的信息泄露到缩放器中。
            train_data = []
            for i in range(0, len(df_data), self.total_seq_len):
                unit = df_data.iloc[i:i + self.total_seq_len]
                subset = unit.iloc[border1s[0]:border2s[0]]  # border1s[0] 和 border2s[0] 定义了训练集的范围
                train_data.append(subset)
            train_data = pd.concat(train_data, axis=0, ignore_index=True)
            self.scaler.fit(train_data.values)

            # 4. 使用拟合好的缩放器转换整个数据集。
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        #从缩放后的数据中从当前flag提取相应的数据段
        data_split = []
        stamp_split = []
        for i in range(0, len(data), self.total_seq_len):
            unit = data[i:i + self.total_seq_len]
            subset = unit[border1:border2]
            data_split.append(subset)

            stamp_unit = data_stamp[i:i + self.total_seq_len]
            stamp_subset = stamp_unit[border1:border2]
            stamp_split.append(stamp_subset)

        data_split = np.concatenate(data_split, axis=0)
        stamp_split = np.concatenate(stamp_split, axis=0)

        self.data_x = data_split
        self.data_y = data_split

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = stamp_split

    def __getitem__(self, index):
        #必须为seq_y引入label机制 itransformer没有使用到seq_y，但是这里是多个模型，必须修改
        num_windows_per_series = (self.interval - self.seq_len - self.pred_len +1)
        seq_id = index // num_windows_per_series
        seq_idx = index % num_windows_per_series

        s_begin = seq_id * self.interval + seq_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin+self.label_len+self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        """ 
        由于在叮咚数据集上进行选段拼接 原有的代码逻辑被修改
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end] 
        """

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        num_series = len(self.data_x) // self.interval
        windows_per_series = (self.interval - self.seq_len - self.pred_len + 1)
        return windows_per_series * num_series

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Pred(Dataset):
    """
    用于预测任务的数据集类。
    它加载训练数据的末尾部分和评估数据的起始部分，以构建一个或多个用于生成未来预测的输入序列。
    这个类的逻辑与新的 Dataset_Custom 类保持兼容。
    """
    def __init__(
        self,
        args=None,      # 兼容性参数，当前未使用
        root_path=None, # 兼容性参数，当前未使用
        flag='pred',
        size=None,
        total_seq_len=90, # 每个独立时间序列的总历史长度
        features='MS',
        data_path=None,   # 预测时可以使用恢复后的数据路径
        target='sale_amount',
        scale=True,
        timeenc=1,        # 与 Dataset_Custom 保持一致
        freq='d',
        seasonal_patterns=None, # 兼容性参数，当前未使用
    ):
        # --- 1. 初始化基本参数 ---
        self.args = args
        self.total_seq_len = total_seq_len

        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 30
            self.label_len = 7  # 默认与 pred_len 保持一致
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # 确保 flag 正确
        assert flag == 'pred', "Dataset_Pred class is only for 'pred' flag."

        # 初始化其他属性
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = data_path
        
        # --- 2. 读取和处理数据 ---
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # --- 核心修正：根据 data_path 智能加载数据 ---
        if self.data_path:
            # 如果提供了 data_path，说明我们要用恢复后的需求数据
            df_train = pd.read_parquet(self.data_path)
        else:
            # 否则，加载原始数据
            dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            df_train = dataset['train'].to_pandas()

        # 评估集总是从原始数据加载，因为它包含真实的未来标签
        dataset_eval = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        df_eval = dataset_eval['eval'].to_pandas()

        # --- 合并数据以创建连续序列 ---
        df_eval = df_eval.sort_values(by='dt')
        future_dates = df_eval['dt'].unique()[:self.pred_len]
        df_future = df_eval[df_eval['dt'].isin(future_dates)]

        df_all = pd.concat([df_train, df_future])
        df_all = df_all.rename(columns={'dt': 'date'})
        df_all = df_all.sort_values(by=['store_id', 'product_id', 'date'])

        #为防止StandardScaler利用预测数据进行平均只能使用训练数据集进行scale
        df_train_processed = df_train.rename(columns={'dt': 'date'})
        df_cols_for_fit = ['discount', 'holiday_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level', self.target]
        df_train_for_scaler = df_train_processed[df_cols_for_fit]
        # --- 特征选择和目标列索引 ---
        df_cols = ['discount', 'holiday_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
                   self.target]
        if self.features == 'S':
            df_cols = [self.target]

        try:
            self.target_col_index = df_cols.index(self.target)
        except ValueError:
            raise ValueError(f"目标列 '{self.target}' 在所选特征中未找到!")

        df_data = df_all[df_cols]

        # --- 生成时间戳特征 (seq_x_mark, seq_y_mark) ---
        df_stamp = df_all[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.DatetimeIndex(df_stamp['date']), freq=self.freq).transpose()

        # --- 修改scale逻辑：与源码保持一致 ---
        if self.scale:
            # 直接在df_data上拟合和转换scaler（与源码保持一致）
            self.scaler.fit(df_train_for_scaler.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # --- 提取用于预测的窗口 ---
        # 每个序列的长度是 历史总长 + 预测长度
        series_len = self.total_seq_len + self.pred_len
        # 我们需要提取的窗口是 历史输入长度 + 预测长度
        window_size = self.seq_len + self.pred_len

        data_x_list = []
        data_stamp_list = []
        # 按每个独立的时间序列进行迭代
        for i in range(0, len(data), series_len):
            # 从每个序列的末尾提取出长度为 window_size 的数据
            start_idx = i + series_len - window_size
            end_idx = i + series_len
            data_x_list.append(data[start_idx:end_idx])
            data_stamp_list.append(data_stamp[start_idx:end_idx])

        self.data_x = np.concatenate(data_x_list, axis=0)
        self.data_stamp = np.concatenate(data_stamp_list, axis=0)

    def __getitem__(self, index):
        # 每个样本的长度是 seq_len + pred_len
        sample_len = self.seq_len + self.pred_len

        # 定位样本的起始位置
        s_begin = index * sample_len

        # 定义输入序列（历史）和目标序列（未来）的边界
        s_end = s_begin + self.seq_len
        r_begin = s_end-self.label_len
        r_end = r_begin + self.pred_len+self.label_len

        # --- 切分数据和时间戳 ---
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_y = seq_y.copy()
        seq_y[-self.pred_len:, self.target_col_index] = 0#这里需要注意，代码逻辑发生了变化不是将所有的y都置为零(由于引入label_len)而是仅仅把最后的pred_len的变量置为零

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
    
        return len(self.data_x)//(self.seq_len + self.pred_len)

    def inverse_transform(self, data):
        """
        对标准化后的数据进行逆转换，恢复到原始尺度。
        """
        return self.scaler.inverse_transform(data)
    
class Dataset_PEMS(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="PEMS03.npz",
        target="OT",
        scale=True,
        scaler=None,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        # print("data file:", data_file)
        data = np.load(data_file, allow_pickle=True)
        data = data["data"][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[: int(train_ratio * len(data))]
        valid_data = data[
            int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))
        ]
        test_data = data[int((train_ratio + valid_ratio) * len(data)) :]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = (
            df.fillna(method="ffill", limit=len(df))
            .fillna(method="bfill", limit=len(df))
            .values
        )

        self.data_x = df
        self.data_y = df

        # generate synthetic data stamp with every 5 minutes in the date format
        data_stamp = pd.date_range(
            start="1/1/2012", periods=len(df), freq="5min"
        ).to_series()
        # data_stamp = data_stamp.drop(0, axis=1)
        # data_stamp = data_stamp.values
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type == 2:
            s_begin = index * 12
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // 12
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
