"""
Copyright (C) 2021 Shivam Pandey

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from zipfile import ZipFile

import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler

import torch
from torch.utils.data import Dataset


from kaggle.api.kaggle_api_extended import KaggleApi


def split_datetime(data):
    tmp = data.copy()
    tmp['year'] = pd.Series(data.index.year.values, index=data.index)
    tmp['month'] = pd.Series(data.index.month.values, index=data.index)
    tmp['day'] = pd.Series(data.index.day.values, index=data.index)
    tmp['hrs'] = pd.Series(data.index.hour.values, index=data.index)
    tmp['mins'] = pd.Series(data.index.minute.values, index=data.index)

    return tmp


class ElectricLoad:

    def __init__(self, data_dir="../../data"):
        self.data_dir = data_dir
        self.dataset_url = 'shivampr21/electricityload'
        self.data_path = os.path.join(self.data_dir, 'electricity-load.csv')
        self.data = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        if not os.path.exists(self.data_path):
            print("{} not found, the data will be downloaded in the specified folder {}".format(self.data_path,
                                                                                                self.data_dir))
            self.api = KaggleApi()
            self.api.authenticate()
            self.download()
        else:
            print("Data found in the given directory {}".format(self.data_dir))
            self.read()

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.api.dataset_download_files(self.dataset_url,
                                        path=self.data_dir)
        zf = ZipFile(os.path.join(self.data_dir, 'electricityload.zip'))
        zf.extractall(self.data_dir)
        zf.close()
        os.remove(os.path.join(self.data_dir, 'electricityload.zip'))
        self.read()

    def read(self):
        print("Reading data at {}".format(self.data_path))
        self.data = pd.read_csv(self.data_path, delimiter=',', parse_dates=['datetime', 'date'], index_col=0, header=0)

    def process_data(self, train_val_split=0.9):
        mask = self.data['load'].isna()

        # Prediction Split
        self.test_data = self.data[mask]
        self.test_data = self.test_data.sort_values(by="datetime")

        # Useful data
        useful_data = self.data.drop(self.test_data.index)

        # train split
        self.train_data = useful_data.sample(frac=train_val_split)
        self.train_data = self.train_data.sort_values(by="datetime")
        # validation data
        self.val_data = useful_data.drop(self.train_data.index)
        self.val_data = self.val_data.sort_values(by="datetime")

        self.test_data = split_datetime(self.test_data.set_index(['datetime']))
        self.train_data = split_datetime(self.train_data.set_index(['datetime']))
        self.val_data = split_datetime(self.val_data.set_index(['datetime']))

        print(
            "Data Split and Processing Done \n "
            "Train Data Size : {} \n Validation Data Size : {} \n Test Data Size : {}".format(
                self.train_data.shape, self.val_data.shape, self.test_data.shape))

        print("Available features in the data: {}".format(self.train_data.columns))

    def normaize(self, method='robust'):

        scalar = None
        if method == 'min-max':
            scalar = MinMaxScaler()
        elif method == 'max-abs':
            scalar = MaxAbsScaler()
        elif method == 'standard':
            scalar = StandardScaler()
        else:
            scalar = RobustScaler()



class ElectricLoadDataset(Dataset):

    def __init__(self, data_inst, window_size=12, input_other_features=False, output_other_features=False,
                 usage='train', transform=None, target_transform=None):
        if not isinstance(data_inst, ElectricLoad):
            raise "Param data_inst should be of type {}, but given of type {}".format(type(pd.DataFrame()),
                                                                                      type(data_inst))

        if usage == 'train':
            self.data = data_inst.train_data
        elif usage == 'test':
            self.data = data_inst.test_data
        elif usage == 'val':
            self.data = data_inst.val_data
        else:
            raise Exception("Param \"data_inst\" should be one of the \"train\"  \"test\" \"val\" ")

        if input_other_features:
            self.input_size = 11
        else:
            self.input_size = 5

        if output_other_features:
            self.output_size = 7
        else:
            self.output_size = 1

        self.window_size = window_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx * self.window_size
        print('window: {}-{}'.format(idx, idx + self.window_size - 1))
        data = torch.zeros(self.window_size, self.input_size)
        target = torch.zeros(self.window_size, self.output_size)

        for i in range(0, self.window_size):
            data_in = self.data.iloc[idx + i][['year', 'month', 'day', 'hrs', 'mins']]
            if self.input_size == 11:
                data_in = self.data.iloc[idx + i][['year', 'month', 'day', 'hrs', 'mins',
                                                   'load', 'apparent_temperature', 'temperature',
                                                   'humidity', 'dew_point', 'wind_speed', 'cloud_cover']]

            data_out = self.data.iloc[idx + i][['load']]
            if self.output_size == 7:
                data_out = self.data.iloc[idx + i][['load', 'apparent_temperature', 'temperature',
                                                    'humidity', 'dew_point', 'wind_speed', 'cloud_cover']]

            data[i] = torch.tensor(data_in)
            target[i] = torch.tensor(data_out)

        return data, target
