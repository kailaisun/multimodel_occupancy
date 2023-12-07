import os
import pandas as pd
from datetime import datetime, timedelta
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from PIL import Image
import librosa
import torch
from transformers import Wav2Vec2Processor
import numpy as np
from torchvision import transforms
from PIL import Image

import torchaudio.transforms as T
# Preprocessing function
def preprocess_and_save2(audio_root, image_root, label_root, csv_output_path):
    records = []
    # Process labels and paths
    # Process labels first to avoid unnecessary file loading
    CT=0
    for zone_label_dir in os.listdir(label_root):
        zone_label_path = os.path.join(label_root, zone_label_dir)
        for label_file in os.listdir(zone_label_path):
            label_path = os.path.join(zone_label_path, label_file)

            # 读取CSV文件
            label_data = pd.read_csv(label_path)
            label_data['timestamp'] = pd.to_datetime(label_data['timestamp'])

            # 创建完整的时间索引，并向前填充缺失值
            start_time = label_data['timestamp'].min()
            end_time = label_data['timestamp'].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            full_index = pd.date_range(start=start_time, end=end_time, freq='S')
            label_data.set_index('timestamp', inplace=True)
            label_data = label_data.reindex(full_index)
            label_data['occupied'].ffill(inplace=True)

            # 重置索引，以便时间戳作为一列
            label_data.reset_index(inplace=True)
            label_data.rename(columns={'index': 'timestamp'}, inplace=True)

            # 处理10秒窗口
            # 初始化一个列表来存储新标签
            new_labels = []
            for index in range(0, len(label_data), 10):
                window = label_data.iloc[index:index + 10]
                label = 1 if window['occupied'].sum() > 0 else 0
                new_labels.append(label)
            # 只保留每个窗口的第一行数据
            processed_data = label_data.iloc[::10].copy()
            # 将新标签列添加到处理后的数据中
            processed_data['new_occupied'] = new_labels

            # 转换为DataFrame
            processed_df = pd.DataFrame(processed_data)

            # 继续处理...

            # 分别筛选出标签为1和0的行
            occupied_df = processed_df[processed_df['occupied'] == 1]
            non_occupied_df = processed_df[processed_df['occupied'] == 0]

            # 计算两种标签的数量，选择较小的那个作为采样数量
            min_count = min(len(occupied_df), len(non_occupied_df))



            # 从两种类型的数据中各自随机选择min_count数量的样本
            sampled_occupied = occupied_df.sample(n=min_count)
            sampled_non_occupied = non_occupied_df.sample(n=min_count)

            # 合并这两部分数据，并随机打乱顺序
            balanced_df = pd.concat([sampled_occupied, sampled_non_occupied]).sample(frac=1).reset_index(drop=True)
            print('balanced_df:',len(balanced_df))

            # 接下来的处理...
            for index, row in balanced_df.iterrows():
                    # Construct the corresponding audio and image file names
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%d_%H%M%S')
                    audio_file = f"{timestamp_str}_RS2_H1.csv"
                    image_file = f"{timestamp_str}_RS2_H1.png"

                    # Construct file paths
                    audio_path = os.path.join(audio_root, 'H1_RS2_AUDIO',
                                              audio_file[0:10], audio_file[11:15], audio_file)
                    image_path = os.path.join(image_root, 'H1_RS2_IMAGES',
                                              image_file[0:10], image_file[11:15], image_file)

                    # Check if the corresponding audio and image files exist
                    if os.path.exists(audio_path) and os.path.exists(image_path):
                        CT=CT+1
                        if CT==10:
                            CT=0
                            records.append({
                                'audio_path': audio_path,
                                'image_path': image_path,
                                'label': row['new_occupied']
                            })
                        else:
                            continue
            print('records:',len(records))
    # Save the records to a CSV file
    df = pd.DataFrame(records, columns=['audio_path', 'image_path', 'label'])
    df.to_csv(csv_output_path, index=False)
    print(f"Preprocessing completed and data saved to {csv_output_path}")

def preprocess_and_save(audio_root, image_root, label_root, csv_output_path):
    records = []
    # Process labels and paths
    # Process labels first to avoid unnecessary file loading
    for zone_label_dir in os.listdir(label_root):
        zone_label_path = os.path.join(label_root, zone_label_dir)
        for label_file in os.listdir(zone_label_path):
            label_path = os.path.join(zone_label_path, label_file)

            # 读取CSV文件
            label_data = pd.read_csv(label_path)
            label_data['timestamp'] = pd.to_datetime(label_data['timestamp'])

            # 创建完整的时间索引，并向前填充缺失值
            start_time = label_data['timestamp'].min()
            end_time = label_data['timestamp'].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            full_index = pd.date_range(start=start_time, end=end_time, freq='S')
            label_data.set_index('timestamp', inplace=True)
            label_data = label_data.reindex(full_index)
            label_data['occupied'].ffill(inplace=True)

            # 重置索引，以便时间戳作为一列
            label_data.reset_index(inplace=True)
            label_data.rename(columns={'index': 'timestamp'}, inplace=True)

            # 处理10秒窗口
            # 初始化一个列表来存储新标签
            new_labels = []
            for index in range(0, len(label_data), 10):
                window = label_data.iloc[index:index + 10]
                label = 1 if window['occupied'].sum() > 0 else 0
                new_labels.append(label)
            # 只保留每个窗口的第一行数据
            processed_data = label_data.iloc[::10].copy()
            # 将新标签列添加到处理后的数据中
            processed_data['new_occupied'] = new_labels

            # 转换为DataFrame
            processed_df = pd.DataFrame(processed_data)

            # 继续处理...

            # 分别筛选出标签为1和0的行
            occupied_df = processed_df[processed_df['occupied'] == 1]
            non_occupied_df = processed_df[processed_df['occupied'] == 0]

            # 计算两种标签的数量，选择较小的那个作为采样数量
            min_count = min(len(occupied_df), len(non_occupied_df))

            # print(len(occupied_df), len(non_occupied_df))
            # continue
            # 从两种类型的数据中各自随机选择min_count数量的样本
            sampled_occupied = occupied_df.sample(n=min_count)
            sampled_non_occupied = non_occupied_df.sample(n=min_count)

            # 合并这两部分数据，并随机打乱顺序
            balanced_df = pd.concat([sampled_occupied, sampled_non_occupied]).sample(frac=1).reset_index(drop=True)
            print('balanced_df:',len(balanced_df))

            # 接下来的处理...
            for index, row in balanced_df.iterrows():
                    # Construct the corresponding audio and image file names
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%d_%H%M%S')
                    audio_file = f"{timestamp_str}_{zone_label_dir[-14:-11]}_H1.csv"
                    image_file = f"{timestamp_str}_{zone_label_dir[-14:-11]}_H1.png"

                    # Construct file paths
                    audio_path = os.path.join(audio_root, zone_label_dir.replace('ZONELABELS', 'AUDIO'),
                                              audio_file[0:10], audio_file[11:15], audio_file)
                    image_path = os.path.join(image_root, zone_label_dir.replace('ZONELABELS', 'IMAGES'),
                                              image_file[0:10], image_file[11:15], image_file)

                    # Check if the corresponding audio and image files exist
                    if os.path.exists(audio_path) and os.path.exists(image_path):
                        records.append({
                            'audio_path': audio_path,
                            'image_path': image_path,
                            'label': row['new_occupied']
                        })
            print('records:',len(records))
    # Save the records to a CSV file
    df = pd.DataFrame(records, columns=['audio_path', 'image_path', 'label'])
    df.to_csv(csv_output_path, index=False)
    print(f"Preprocessing completed and data saved to {csv_output_path}")

def presave(path='output.csv'):
    import pandas as pd
    import shutil
    import os

    # Load the CSV file
    df = pd.read_csv(path)  # Replace with your CSV file path

    # Create a new folder if it doesn't exist
    new_folder = 'datamerged'
    os.makedirs(new_folder, exist_ok=True)

    # Function to copy files to the new folder and return the new path
    def copy_file(old_path):
        file_name = os.path.basename(old_path)
        new_path = os.path.join(new_folder, file_name)
        shutil.copy2(old_path, new_path)
        return new_path

    def copy_file2(old_path):
        file_name = os.path.basename(old_path)
        for i in range(10):
            file_name = file_name[0:16]+str(i)+file_name[17:]
            new_path = os.path.join(new_folder, file_name)
            if i == 0:pathcc=new_path
            # shutil.copy2(old_path, new_path)
        return pathcc

    # Apply the function to audio and image columns
    df['new_audio_path'] = df.iloc[:, 0].apply(copy_file)
    df['new_image_path'] = df.iloc[:, 1].apply(copy_file2)

    # Save the new paths to a CSV file
    df.to_csv('output_2.csv', index=False)


# Custom Dataset class that reads from a CSV file
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.envmin=[0,0,0,3,400,35186,35501]

        self.envmax=[2990,28.1,100,10812,16953,35212,36775]


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        audio_path = self.data_info.iloc[idx]['new_audio_path']
        image_path = self.data_info.iloc[idx]['new_image_path']
        label = self.data_info.iloc[idx]['label']



        # Load the CSV file
        # print(audio_path)
        data = pd.read_csv(audio_path.replace('\\', '/'))
        # Load the model
        # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # Preprocess the data
        # waveform = data.iloc[:, 0].to_numpy()  # Convert the first column of dataframe to numpy array
        # waveform = waveform / np.max(np.abs(waveform))  # Normalize the waveform
        # waveform=torch.from_numpy(waveform)

        audio_waveform = data.iloc[:, 0].values
        last_sample = audio_waveform[-1]
        audio_waveform = np.append(audio_waveform, last_sample)
        audio_waveform =2* (audio_waveform - audio_waveform.min()) / (audio_waveform.max()-audio_waveform.min())-1

        required_length = 80000  # 16kHz采样率下1秒的长度
        if len(audio_waveform) < required_length:
            audio_waveform_16k = np.interp(
                np.linspace(0, 1, required_length),
                np.linspace(0, 1, len(audio_waveform)),
                audio_waveform
            )

        audio_waveform_16k = librosa.resample(audio_waveform_16k, orig_sr=8000, target_sr=16000)


        # resampler = T.Resample(8000, 16000, dtype=torch.float64)
        # resampled_waveform = resampler(waveform)

        # Use the correct sampling rate (8 kHz)
        # waveform = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        image_path = image_path.replace('\\', '/')
        images=[]
        for ii in range(10):
            im_path = image_path[0:27] + str(ii) + image_path[28:]
            image = Image.open(im_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            # Define the transform to convert PIL images to tensors
        # transform_to_tensor = transforms.ToTensor()
        #     # Apply your existing transforms, if any, then convert to tensor
        # image = transform_to_tensor(image)


        env_path='../H1_ENVIRONMENTAL/H1_RS2_ENV/'+audio_path[11:21]+audio_path[-11:-4]+'_env.csv'
        env=pd.read_csv(env_path, header=None)
        env[0] = pd.to_datetime(env[0]).dt.strftime('%Y-%m-%d_%H%M%S')

        env.fillna(method='ffill', inplace=True)
        matched_rows = env[env[0] == audio_path[11:28]]
        if matched_rows.empty:
            matched_rows = env[env[0] == audio_path[11:26]+'00']

        env_data = matched_rows.iloc[0, 1:8].values
        env_data = pd.to_numeric(env_data, errors='coerce')

        # 转换为 PyTorch Tensor
        env_tensor = torch.tensor(env_data, dtype=torch.float32)
        env_tensor = 2*(env_tensor - torch.tensor(self.envmin)) / (torch.tensor(self.envmax) - torch.tensor(self.envmin))-1
        return audio_waveform_16k, images, env_tensor,label

# Example usage of the preprocessing function
# preprocess_and_save('/path/to/audio', '/path/to/images', '/path/to/labels', 'process.csv')

# Example usage of the dataset class
# dataset = MultimodalDataset('process.csv', transform=your_transforms)
