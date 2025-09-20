import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

from utils.augmentations import Injector

def create_cmapss_labels(df, rul_threshold=130):
    # 根据每个单元的最大周期计算RUL
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().to_dict()
    df['RUL'] = df.apply(lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], axis=1)
    df['label'] = (df['RUL'] <= rul_threshold).astype(int)
    return df['label'].values
    
def get_dataset(args, domain_type, split_type):
    """
    Return the correct dataset object that will be fed into dataloader
    (最终修复版：根据 domain_type 选择正确的路径来判断数据集)
    """

    if domain_type == "source":
        path_to_check = args.path_src
    else: # domain_type is "target"
        path_to_check = args.path_trg

    normalized_path = os.path.normpath(path_to_check)
    dataset_name = os.path.basename(normalized_path)

    print(f"Loading '{domain_type}' data from folder '{dataset_name}'...")

    # --- 数据集选择逻辑 ---
    if dataset_name == "SMD":
        if domain_type == "source":
            return SMDDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SMDDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)
            
    elif dataset_name == "SWaT": # 匹配 'SWaT' 文件夹
        if domain_type == "source":
            return SWAT_1D_Dataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SWAT_1D_Dataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)       
    elif dataset_name == "MSL_SMAP":
        if domain_type == "source":
            # 如果MSL作为源域，也使用2D
            return MSL_2D_Dataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else: # "target"
            # 当MSL作为目标域时，调用新的2D类
            return MSL_2D_Dataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)
    elif dataset_name == "MBA":
        if domain_type == "source":
            return MBADataset(args.path_src, subject_id=None, split_type=split_type, is_cuda=True)
        else: # "target"
            return MBADataset_trg(args.path_trg, subject_id=None, split_type=split_type, is_cuda=True)
    elif dataset_name == "CATS":
        if domain_type == "source":
            return CATSDataset(args.path_src, subject_id=None, split_type=split_type, is_cuda=True)
        else: # "target"
            return CATSDataset_trg(args.path_trg, subject_id=None, split_type=split_type, is_cuda=True)
    elif dataset_name == "C-MAPSS":
        if domain_type == "source":
            # C-MAPSS 作为源域时，subject_id 来自 id_src
            return CMAPSSDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else: # "target"
            # C-MAPSS 作为目标域时，subject_id 来自 id_trg
            return CMAPSSDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True) 
    elif dataset_name == "PHM08":
        if domain_type == "source":
            return PHM08Dataset(args.path_src, split_type=split_type, is_cuda=True)
        else: # "target"
            return PHM08Dataset_trg(args.path_trg, split_type=split_type, is_cuda=True)
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}' extracted from path '{path_to_check}'. Please check your folder names and get_dataset function.")

class MSLDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose

        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)

        # self.mean = None
        # self.std = None
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}

        return sample

    def load_sequence(self):
        path_sequence = os.path.join(self.root_dir, f"{self.subject_id}.npy")
        path_label = os.path.join(self.root_dir, f"{self.subject_id}_labels.npy")
        
        sequence_data = np.load(path_sequence)
        labels = np.load(path_label)

        # --- 核心修改：特征选择 ---
        print(f"Original MSL ({self.subject_id}) feature dim: {sequence_data.shape[1]}")
        if sequence_data.shape[1] > 24: # 只在需要时进行降维
            selector = SelectKBest(score_func=f_classif, k=24)
            sequence_data = selector.fit_transform(sequence_data, labels)
            print(f"MSL ({self.subject_id}) feature dim after selection: {sequence_data.shape[1]}")

        # 归一化 
        scaler = StandardScaler()
        self.sequence = scaler.fit_transform(sequence_data)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        self.label = labels

        # 滑动窗口
        wsz, stride = 100, 1 
        self.sequence, self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        self.mean = np.mean(self.sequence, axis=0)
        self.std = np.std(self.sequence, axis=0)
        self.std[self.std==0.0] = 1.0
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class CMAPSSDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id # e.g., 'FD001'
        self.split_type = split_type # 'train' or 'test'
        self.is_cuda = is_cuda
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def load_sequence(self):
        print(f"INFO: Loading C-MAPSS data for '{self.split_type}' split (using subject: {self.subject_id})...")

        # 1. 加载数据
        load_split_type = 'train' if self.split_type == 'train' else 'test'
        path = os.path.join(self.root_dir, f'{load_split_type}_{self.subject_id}.txt')
        print(f"DEBUG: Attempting to load C-MAPSS data from: {path}")

        if not os.path.exists(path):
            print(f"ERROR: Cannot find data file at {path}. Initializing as empty dataset.")
            # 初始化为空，防止崩溃
            self.sequence, self.label, self.positive, self.negative = [np.array([]) for _ in range(4)]
            self.mean, self.std = np.array([]), np.array([])
            return

        df = pd.read_csv(path, sep=' ', header=None)
        df.drop(columns=[26, 27], inplace=True) # 移除末尾的空列
        column_names = ['unit_number', 'time_in_cycles'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        df.columns = column_names

        # 2. 生成标签
        print("INFO: Generating labels based on RUL...")
        if load_split_type == 'train':
            # 训练集有完整的寿命周期，可以直接计算RUL并生成标签
            labels = create_cmapss_labels(df)
        else: # 'test'
            rul_path = os.path.join(self.root_dir, f'RUL_{self.subject_id}.txt')
            df_rul = pd.read_csv(rul_path, header=None)
            df['RUL'] = df_rul[0] 
            labels = create_cmapss_labels(df) 

        # 3. 特征选择 (!!! 关键对齐步骤 !!!)
        all_features = [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
        
        selector = SelectKBest(score_func=f_classif, k=2)
        features_2d = selector.fit_transform(df[all_features], labels)
        selected_mask = selector.get_support()
        selected_names = [name for name, selected in zip(all_features, selected_mask) if selected]
        print(f"INFO: Original C-MAPSS feature count: {len(all_features)}. Selected features: {selected_names}")

        # 4. 数据归一化 (C-MAPSS 通常使用 MinMaxScaler)
        scaler = MinMaxScaler()
        self.sequence = scaler.fit_transform(features_2d)
        self.mean = scaler.data_min_ # 保存统计量
        self.std = scaler.data_max_ - scaler.data_min_ # 保存统计量
        self.label = labels
        print(f"INFO: Loaded C-MAPSS data. Shape after feature selection: {self.sequence.shape}")

        # 5. 滑动窗口处理
        wsz, stride = 100, 1
        self.sequence, self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]
        print(f"INFO: C-MAPSS data converted to {len(self.sequence)} windows.")


    def convert_to_windows(self, w_size, stride):
        windows, wlabels = [], []
        sz = (self.sequence.shape[0] - w_size) // stride + 1
        for i in range(sz):
            st = i * stride
            ed = st + w_size
            w = self.sequence[st:ed]
            lbl = 1 if np.any(self.label[st:ed]) else 0
            windows.append(w)
            wlabels.append(lbl)
        return np.array(windows), np.array(wlabels)

    def get_statistic(self):
        return self.mean, self.std

    def __getitem__(self, id_):
        # 源域的采样逻辑，与 MBA/CATS 保持一致
        sequence = self.sequence[id_]
        label = self.label[id_]
        if len(self.positive) > 0:
            pid_ = np.random.randint(0, len(self.positive))
            positive = self.positive[pid_]
        else:
            positive = sequence.copy()
        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            positive = torch.Tensor(positive).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else: # CPU
            sequence, positive, negative = [torch.Tensor(x).float() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class CMAPSSDataset_trg(CMAPSSDataset):
    # 目标域的采样逻辑，与 MBA/CATS 保持一致
    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        pid_ = min(max(0, pid_), len(self.sequence) - 1)
        positive = self.sequence[pid_]
        negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence, positive, negative = [torch.Tensor(x).float().cuda() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            label = torch.Tensor([label]).long().cuda()
        else: # CPU
            sequence, positive, negative = [torch.Tensor(x).float() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class MBADataset(Dataset):
    def __init__(self, root_dir, subject_id=None, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.subject_id = subject_id 
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def load_sequence(self):
        load_split_type = self.split_type
        if self.split_type == 'val':
            print("INFO: 'val' split requested. Checking for 'val.csv' or using 'test.csv' as fallback.")
            # 检查 val.csv 是否真的存在
            val_path = os.path.join(self.root_dir, "val.csv")
            if not os.path.exists(val_path):
                print("INFO: 'val.csv' not found. Using 'test.csv' for validation.")
                load_split_type = 'test'
            else:
                print("INFO: Found 'val.csv'. Using it for validation.")

        data_path = os.path.join(self.root_dir, f"{load_split_type}.csv") 
        label_path = os.path.join(self.root_dir, "labels.csv")

        # [新增调试信息] 明确打印将要加载的文件路径
        print(f"DEBUG: Attempting to load data from: {data_path}")
        print(f"DEBUG: Attempting to load labels from: {label_path}")
        
        try:
            # 优先尝试 gbk 编码
            df_data = pd.read_csv(data_path, skiprows=[1], encoding='gbk')
            df_label = pd.read_csv(label_path, encoding='gbk')
            print("INFO: Successfully read CSV files with 'gbk' encoding.")
        except Exception:
            # 如果失败，尝试 utf-8
            print("INFO: 'gbk' failed. Trying 'utf-8' encoding...")
            df_data = pd.read_csv(data_path, skiprows=[1], encoding='utf-8')
            df_label = pd.read_csv(label_path, encoding='utf-8')
            print("INFO: Successfully read CSV files with 'utf-8' encoding.")

        features = df_data[['ECG1', 'ECG2']].values
        print(f"INFO: Reading labels for '{load_split_type}' split...")
        
        labels = np.zeros(len(features), dtype=int)
        raw_anomaly_indices = df_label['Sample'].values
        
        print("INFO: Applying anomaly window expansion (like TranAD)...")
        expansion_window = 20
        for idx in raw_anomaly_indices:
            if pd.isna(idx): continue
            idx = int(idx)
            start = max(0, idx - expansion_window)
            end = min(len(labels), idx + expansion_window + 1)
            labels[start:end] = 1
        
        self.label = labels

        scaler = StandardScaler()
        self.sequence = scaler.fit_transform(features)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        print(f"INFO: Loaded MBA data from {data_path}. Shape: {self.sequence.shape}")
        
        wsz, stride = 100, 1
        self.sequence, self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]
        print(f"INFO: Data converted to {len(self.sequence)} windows.")
        if len(self.positive) == 0:
            print("WARNING: No positive (normal) windows found!")
        if len(self.negative) == 0:
            print("WARNING: No negative (anomaly) windows found!")

    def convert_to_windows(self, w_size, stride):
        windows, wlabels = [], []
        sz = (self.sequence.shape[0] - w_size) // stride + 1
        for i in range(sz):
            st = i * stride
            ed = st + w_size
            w = self.sequence[st:ed]
            lbl = 1 if np.any(self.label[st:ed]) else 0
            windows.append(w)
            wlabels.append(lbl)
        return np.array(windows), np.array(wlabels)

    def get_statistic(self):
        return self.mean, self.std

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        if len(self.positive) > 0:
            pid_ = np.random.randint(0, len(self.positive))
            positive = self.positive[pid_]
        else:
            positive = sequence.copy() 
        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            positive = torch.Tensor(positive).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class MBADataset_trg(MBADataset):
    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        pid_ = min(max(0, pid_), len(self.sequence) - 1)
        positive = self.sequence[pid_]
        negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            positive = torch.Tensor(positive).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class MSL_2D_Dataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

    def load_sequence(self):
        with open(os.path.join(self.root_dir, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')
        data_info = csv_reader[csv_reader['chan_id'] == self.subject_id]
        path_sequence = os.path.join(self.root_dir, 'test/', str(self.subject_id) + ".npy")
        temp = np.load(path_sequence)
        
        # --- [核心修改]: 将数据降至2维 ---
        if temp.shape[1] > 2:
            print(f"Original MSL dimension was {temp.shape[1]}. Reducing to 2.")
            temp = temp[:, :2] 
        
        print(f"Domain (MSL_2D) feature dimension: {temp.shape[1]}")

        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)
        
        # 使用 StandardScaler 进行归一化
        scaler = StandardScaler()
        self.sequence = scaler.fit_transform(temp)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        self.std[self.std==0.0] = 1.0 # 防止除以0
        
        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.label = np.asarray(labels)
        wsz, stride = 100, 1
        self.sequence , self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def get_statistic(self):
        return self.mean, self.std

    def convert_to_windows(self, w_size, stride):
        windows, wlabels = [], []
        sz = int((self.sequence.shape[0] - w_size) / stride)
        for i in range(sz):
            st = i * stride
            w = self.sequence[st:st + w_size]
            lbl = 1 if np.any(self.label[st:st + w_size]) > 0 else 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class MSL_2D_Dataset_trg(MSL_2D_Dataset):
    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        pid_ = min(max(0, pid_), len(self.sequence) - 1)
        positive = self.sequence[pid_]
        negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class CATSDataset(Dataset):
    def __init__(self, root_dir, subject_id=None, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.subject_id = subject_id
        # 初始化空的属性，以防加载失败
        self.sequence = np.array([])
        self.label = np.array([])
        self.positive = np.array([])
        self.negative = np.array([])
        self.mean = np.array([])
        self.std = np.array([])
        # 调用加载函数
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def load_sequence(self):
        """
        [ 核心修正区域 ]
        增加了对 val/test 文件不存在的健壮性处理。
        """
        print(f"INFO: Loading CATS data for '{self.split_type}' split...")
        
        load_split_type = self.split_type
        # 健壮地处理 'val' split ---
        if self.split_type == 'val':
            print("INFO: 'val' split requested. Checking for 'val.csv' or 'test.csv'.")
            val_path = os.path.join(self.root_dir, "val.csv")
            test_path = os.path.join(self.root_dir, "test.csv")
            if os.path.exists(val_path):
                print("INFO: Found 'val.csv'. Using it for validation.")
                load_split_type = 'val'
            elif os.path.exists(test_path):
                print("INFO: 'val.csv' not found. Using 'test.csv' for validation.")
                load_split_type = 'test'
            else:
                # 如果 val 和 test 都没有，则放弃加载
                print("WARNING: Neither 'val.csv' nor 'test.csv' found. Skipping validation set.")
                return # 提前返回，此时 self.sequence 依然是空数组

        data_path = os.path.join(self.root_dir, f"{load_split_type}.csv")
        print(f"DEBUG: Attempting to load CATS data from: {data_path}")

        # 如果连 train.csv 都找不到，也安全退出
        if not os.path.exists(data_path):
            print(f"ERROR: Cannot find data file at {data_path}. Initializing as empty dataset.")
            return

        df_data = pd.read_csv(data_path)
        print("INFO: Successfully read CATS data file.")

        # 特征选择
        selected_features = ['cfo1', 'cso1']
        print(f"INFO: Selecting features: {selected_features}")
        features = df_data[selected_features].values
        
        # 加载标签
        print("INFO: Reading labels from 'anomaly_label' column...")
        labels = df_data['anomaly_label'].values
        self.label = np.where(labels > 0, 1, 0)

        # 数据归一化
        scaler = StandardScaler()
        self.sequence = scaler.fit_transform(features)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        print(f"INFO: Loaded CATS data. Shape after feature selection: {self.sequence.shape}")
        
        # 滑动窗口处理
        wsz, stride = 100, 1
        self.sequence, self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]
        print(f"INFO: CATS data converted to {len(self.sequence)} windows.")
        if len(self.positive) == 0:
            print("WARNING: No positive (normal) windows found in CATS data!")
        if len(self.negative) == 0:
            print("WARNING: No negative (anomaly) windows found in CATS data!")

    def convert_to_windows(self, w_size, stride):
        windows, wlabels = [], []
        sz = (self.sequence.shape[0] - w_size) // stride + 1
        for i in range(sz):
            st = i * stride
            ed = st + w_size
            w = self.sequence[st:ed]
            lbl = 1 if np.any(self.label[st:ed]) else 0
            windows.append(w)
            wlabels.append(lbl)
        return np.array(windows), np.array(wlabels)

    def get_statistic(self):
        return self.mean, self.std

    def __getitem__(self, id_):
        # 源域采样逻辑
        sequence = self.sequence[id_]
        label = self.label[id_]
        if len(self.positive) > 0:
            pid_ = np.random.randint(0, len(self.positive))
            positive = self.positive[pid_]
        else:
            positive = sequence.copy()
        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            positive = torch.Tensor(positive).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

class CATSDataset_trg(CATSDataset):
    # 目标域采样逻辑
    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        pid_ = min(max(0, pid_), len(self.sequence) - 1)
        positive = self.sequence[pid_]
        negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            positive = torch.Tensor(positive).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

def create_phm_labels(df, rul_threshold=130):
    # 根据每个单元的最大周期计算RUL
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().to_dict()
    df['RUL'] = df.apply(lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], axis=1)
    df['label'] = (df['RUL'] <= rul_threshold).astype(int)
    return df['label'].values

from sklearn.model_selection import train_test_split as sk_train_test_split

class PHM08Dataset(Dataset):
    def __init__(self, root_dir, subject_id=None, split_type="train", is_cuda=True, verbose=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.prepare_data()
        self.load_sequence() # load_sequence现在将使用分割好的数据

    def prepare_data(self):
        """
        一次性加载 train.txt，并将其分割为内部的训练集和验证集。
        """
        print("INFO: Preparing PHM08 data by splitting train.txt...")
        path = os.path.join(self.root_dir, 'train.txt')
        if not os.path.exists(path):
            print(f"ERROR: Cannot find train.txt at {path}")
            self.df_train = pd.DataFrame()
            self.df_val = pd.DataFrame()
            return

        df = pd.read_csv(path, sep=' ', header=None)
        df.dropna(axis=1, how='all', inplace=True)
        num_cols = len(df.columns)
        column_names = ['unit_number', 'time_in_cycles'] + [f'feature_{i}' for i in range(1, num_cols - 1)]
        df.columns = column_names

        # 为分割做准备，确保每个engine的数据不会被切散
        # stratify by unit_number to ensure both splits get a mix of engines
        engine_ids = df['unit_number'].unique()
        train_ids, val_ids = sk_train_test_split(engine_ids, test_size=0.2, random_state=42)

        self.df_train = df[df['unit_number'].isin(train_ids)].copy()
        self.df_val = df[df['unit_number'].isin(val_ids)].copy()
        print(f"INFO: Split complete. {len(train_ids)} engines for training, {len(val_ids)} for validation.")

    def load_sequence(self):
        print(f"INFO: Loading PHM08 data for '{self.split_type}' split...")

        if self.split_type == 'train':
            df = self.df_train
        else: # 'val' or 'test'
            df = self.df_val
        
        if df.empty:
            print(f"WARNING: No data available for split '{self.split_type}'. Initializing as empty dataset.")
            self.sequence, self.label, self.positive, self.negative = [np.array([]) for _ in range(4)]
            return

        all_features = [col for col in df.columns if 'feature' in col]
        labels = create_phm_labels(df)

        print(f"INFO: Aligning to 2 features...")
        selector = SelectKBest(score_func=f_classif, k=2)
        features_2d = selector.fit_transform(df[all_features], labels)
        
        scaler = StandardScaler()
        self.sequence = scaler.fit_transform(features_2d)
        self.mean = scaler.mean_
        self.std = scaler.scale_
        self.label = labels
        
        wsz, stride = 100, 1
        self.sequence, self.label = self.convert_to_windows(wsz, stride)
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]
        print(f"INFO: PHM08 '{self.split_type}' split converted to {len(self.sequence)} windows.")
    
    def __len__(self):
        return len(self.sequence)
        
    def convert_to_windows(self, w_size, stride):
        windows, wlabels = [], []
        sz = (self.sequence.shape[0] - w_size) // stride + 1
        for i in range(sz):
            st = i * stride
            ed = st + w_size
            w = self.sequence[st:ed]
            lbl = 1 if np.any(self.label[st:ed]) else 0
            windows.append(w)
            wlabels.append(lbl)
        return np.array(windows), np.array(wlabels)

    def get_statistic(self):
        return self.mean, self.std

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        if len(self.positive) > 0:
            pid_ = np.random.randint(0, len(self.positive))
            positive = self.positive[pid_]
        else:
            positive = sequence.copy()
        random_choice = np.random.randint(0, 10)
        if random_choice == 0 and len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence, positive, negative = [torch.Tensor(x).float().cuda() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence, positive, negative = [torch.Tensor(x).float() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample
        
class PHM08Dataset_trg(PHM08Dataset):
    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        label = self.label[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        pid_ = min(max(0, pid_), len(self.sequence) - 1)
        positive = self.sequence[pid_]
        negative, _ = get_injector(torch.from_numpy(sequence).float(), self.mean, self.std)
        sequence_mask = np.ones(sequence.shape)
        if self.is_cuda:
            sequence, positive, negative = [torch.Tensor(x).float().cuda() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence, positive, negative = [torch.Tensor(x).float() for x in [sequence, positive, negative]]
            sequence_mask = torch.Tensor(sequence_mask).long()
            label = torch.Tensor([label]).long()
        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

def get_injector(sample, d_mean, d_std):
    # --- 1. 统一输入格式 ---
    # 确保输入是 PyTorch Tensor
    if isinstance(sample, np.ndarray):
        sample = torch.from_numpy(sample).float()
    
    # 确保均值和标准差也是 Tensor
    if isinstance(d_mean, np.ndarray):
        d_mean = torch.from_numpy(d_mean).float()
    if isinstance(d_std, np.ndarray):
        d_std = torch.from_numpy(d_std).float()

    # 检查输入是否是带批次维度的 (如来自CATS的 (1, 100, 17))
    is_batched = sample.ndim == 3
    if not is_batched:
        sample = sample.unsqueeze(0)

    device = sample.device
    d_mean, d_std = d_mean.to(device), d_std.to(device)

    # --- 2. 执行注入逻辑 ---
    # 反标准化
    sample_denorm = (sample * d_std) + d_mean
    
    sample_denorm_2d_np = sample_denorm[0].cpu().numpy()
    
    injected_window = Injector(sample_denorm_2d_np)
    injected_win_denorm_np = injected_window.injected_win
    
    # --- 3. 统一输出格式 ---
    # 将 2D Numpy 结果转换回 Tensor，并恢复批次维度
    injected_win_denorm = torch.from_numpy(injected_win_denorm_np).float().unsqueeze(0).to(device)
    
    # 重新标准化
    injected_win_norm = (injected_win_denorm - d_mean) / d_std
    
    if not is_batched:
        injected_win_norm = injected_win_norm.squeeze(0)

    # 返回Tensor和空标签
    return injected_win_norm, None


def get_output_dim(args):
    output_dim = -1

    if "MSL" in args.path_src:
        output_dim = 1
    elif "CATS" in args.path_src: # 同样使用路径中的关键词 "CATS" 来识别
        output_dim = 1
    else:
        output_dim = 6

    return output_dim

def collate_test(batch):
    #The input is list of dictionaries
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out



