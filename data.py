import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

class CASME2(Dataset):
    def __init__(self,path = '.',
                 img_sz = 128,
                 calculate_strain = False,
                 raw_img = False,
                 initialized_df = None):
        
        if initialized_df is None:
            self.df = pd.read_csv(path + '/' + 'casme2.csv')
            self.df['OpticalFlow'] = None
            self.path = path
            self.img_sz = img_sz
            for idx,row in tqdm(self.df.iterrows(),ascii = '='):
                prefix = self.__get_prefix(row)
                onset = self.__read_img(
                    prefix + str(row['Onset']) + '.jpg'
                )
                apex = self.__read_img(
                    prefix + str(row['Apex']) + '.jpg'
                )
                if raw_img:
                    self.df.at[idx,'OpticalFlow'] = np.vstack(
                        (np.expand_dims(onset.astype(np.float32) / 255, axis = 0),np.expand_dims(apex.astype(np.float32) / 255, axis = 0))
                    )
                else:
                    self.df.at[idx,'OpticalFlow'] = self.__calc_optical_flow(onset,apex)
                    if calculate_strain:
                        self.df.at[idx,'OpticalFlow'] = self.__append_optical_strain(self.df.at[idx,'OpticalFlow'])
                self.df.at[idx,'Class'] = {'negative':0,'positive':1,'surprise':'2'}[row['Class']]  
        else:
            self.df = initialized_df
    
    def __get_prefix(self,row):
        sub_sample = row['Subject'] + '/' + row['Sample'] + '/'
        if row['Dataset'] == 'casme2':
            return self.path + '/casme2_cropped/' + sub_sample + 'reg_img'
        
            
    def __read_img(self,name):
        return cv2.cvtColor(
            cv2.resize(
                cv2.imread(name,cv2.IMREAD_COLOR),
                (self.img_sz,self.img_sz),
                interpolation = cv2.INTER_CUBIC
            ),
            cv2.COLOR_BGR2GRAY
        )
    
    def __calc_optical_flow(self,onset,apex):
        return np.array(cv2.optflow.DualTVL1OpticalFlow_create().calc(onset,apex,None))
    
    def __append_optical_strain(self,flow):
        # need to work on
        return 
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        #return self.df.at[idx,'OpticalFlow'],int(self.df.at[idx,'Class'])
        
class LOSOGenerator():
    def __init__(self,dataset):
        self.data = dataset
        self.subjects = self.data.df[['Dataset','Subject']].drop_duplicates().reset_index()
        self.idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx == len(self.subjects):
            raise StopIteration
        ds,sub = self.subjects.at[self.idx,'Dataset'],self.subjects.at[self.idx,'Subject']
        self.idx += 1
        train_df = self.data.df[(self.data.df.Dataset != ds) | (self.data.df.Subject != sub)] \
            .reset_index(drop=True)
        test_df = self.data.df[(self.data.df.Dataset == ds) & (self.data.df.Subject == sub)] \
            .reset_index(drop=True)
        return CASMECombinedDataset(initialized_df = train_df),CASMECombinedDataset(initialized_df = test_df)