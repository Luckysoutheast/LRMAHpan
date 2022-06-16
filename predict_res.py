import torch
import torch.nn.functional as F
from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
import torch.utils.data as Data
from copy import deepcopy
import copy
from utils import *
import random
import sys
import pickle as pkl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

def hla2seq(df):
    df_cp = copy.deepcopy(df)
    print('df_cp \n',df_cp)
    hlaseq = pd.read_csv('allele_sequences.csv')
    hlaseq['allele'] = hlaseq['allele'].map(lambda x: x.replace(':',''))
    hlaseq['allele'] = hlaseq['allele'].map(lambda x: x.replace('*',''))
    for it in ['A1','A2','B1','B2','C1','C2']:
        df_cp=df_cp.merge(hlaseq,left_on=it, right_on='allele', how='left')
    df_cp = df_cp[['peptide','sequence_x','sequence_y']]
    print('df merge ',df_cp)
    df_cp.columns = ['peptide','A1','B1','C1','A2','B2','C2']
    df_cp['pep_a1'] = df_cp.apply(lambda x: x['peptide'] + x['A1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_a2'] = df_cp.apply(lambda x: x['peptide'] + x['A2'] + x['peptide'][::-1],axis=1)
    df_cp['pep_b1'] = df_cp.apply(lambda x: x['peptide'] + x['B1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_b2'] = df_cp.apply(lambda x: x['peptide'] + x['B2'] + x['peptide'][::-1],axis=1)
    df_cp['pep_c1'] = df_cp.apply(lambda x: x['peptide'] + x['C1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_c2'] = df_cp.apply(lambda x: x['peptide'] + x['C2'] + x['peptide'][::-1],axis=1)
    df_cp = df_cp[['peptide', 'pep_a1', 'pep_a2', 'pep_b1', 'pep_b2', 'pep_c1', 'pep_c2']]
    return df_cp

def HLAData_test_pool(df, word_idx):
    df_ = data_process(df, word_idx)
    testDataset = HLADataset_pool(df_)
    test_loader = DataLoader(dataset = testDataset, batch_size=64, shuffle=False)
    
    return test_loader
def embedding_pool(df):
    tensor_seq = torch.tensor(data=df['seq'])
    onehotseq = F.one_hot(tensor_seq.to(torch.int64), num_classes=22)
    return onehotseq


class HLADataset_pool(Dataset):

  def __init__(self, train_set,transform=None):
    self.train_set = train_set
    self.transform = transform

  def __getitem__(self, index):
    test_inputs = embedding_pool(self.train_set.iloc[index,:])
    if self.transform is not None:
      test_inputs = self.transform(test_inputs)
    test_inputs = test_inputs.float()
    return test_inputs
 
  def __len__(self):
    return len(self.train_set)

def data_process(df, word_idx, max_len=59):
    # df_ = deepcopy(df)
    df_output = pd.DataFrame()
    for a in ['pep_a1','pep_a2','pep_b1','pep_b2','pep_c1','pep_c2']:
        df_output[a] = df[a].map(lambda x: [word_idx[word] for word in x[:max_len]])
        df_output[a] = df_output[a].map(lambda x: x + [0] * (max_len - len(x)))
    df_output['seq'] = df_output.apply(lambda x: np.array([x['pep_a1'],x['pep_a2'],x['pep_b1'],x['pep_b2'],x['pep_c1'],x['pep_c2']]), axis=1)
    return df_output[['seq']]

def DataLoader_test(df, word_idx, device):
    test_inputs = []
    df_process = data_process2(df,word_idx)
    test_inputs = torch.LongTensor(df_process['peptide_enc']).to(device)
    test_dataset = Data.TensorDataset(test_inputs)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    return test_loader

def data_process2(df, word_idx, max_len=14):
    """
    把句子转换为数字序列
    :param sentence:
    :param max_len: 句子的最大长度
    :return:
    """
    df_output = pd.DataFrame()
    df_output['peptide_enc'] = df['peptide'].map(lambda x: [word_idx[word] for word in x[:max_len]])
    df_output['peptide_enc'] = df_output['peptide_enc'].map(lambda x: x + [0] * (max_len - len(x)))
    df_output['peptide_enc'] = df_output['peptide_enc'].map(lambda x: np.array(x))
    return df_output

def ResMHApan_batch2(df, is_save):
    df['A1'] = df['A1'].map(lambda x: x.replace(':','').replace('*',''))
    df['A2'] = df['A2'].map(lambda x: x.replace(':','').replace('*',''))
    df['B1'] = df['B1'].map(lambda x: x.replace(':','').replace('*',''))
    df['B2'] = df['B2'].map(lambda x: x.replace(':','').replace('*',''))
    df['C1'] = df['C1'].map(lambda x: x.replace(':','').replace('*',''))
    df['C2'] = df['C2'].map(lambda x: x.replace(':','').replace('*',''))
    
    # assert len(df.columns) == 7
    print('df: \n',df)
    BA,AP,PS = ResMHApan_predict(df[['peptide','A1','A2','B1','B2','C1','C2']])
    df['BA'] = BA
    df['AP'] = AP
    df['PS'] = PS
    print('final result: \n',df)
    if is_save:
        df.to_csv('./app/download/result.csv',index=False)
    return df

def ResMHApan_predict(df):

    model_paths = ['./BA/1/latest_best_0.9746_160.pth',
    './BA/1/latest_best_0.9748_169.pth',
    './BA/1/latest_best_0.9751_162.pth',
    './BA/2/latest_best_0.9747_160.pth',
    './BA/2/latest_best_0.9750_163.pth',
    './BA/2/latest_best_0.9753_161.pth',
    './BA/3/latest_best_0.9752_160.pth',
    './BA/3/latest_best_0.9755_163.pth',
    './BA/3/latest_best_0.9759_166.pth',
    ]
    ap_model_paths = ['./AP/1/latest_best_0.7912890967553345_123.pth',
    './AP/2/latest_best_0.7946202077250752_123.pth',
    './AP/3/latest_best_0.7935206869633099_103.pth',
    './AP/3/latest_best_0.7902829952416729_110.pth',
    './AP/4/latest_best_0.7911978985974129_107.pth',
    './AP/4/latest_best_0.7960330255508329_109.pth',]
    vocab_list = ['./BA/1/vocal.pkl',
    './BA/1/vocal.pkl',
    './BA/1/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/3/vocal.pkl',
    './BA/3/vocal.pkl',
    './BA/3/vocal.pkl',]
    ap_vocab_list = [    
        './AP/1/vocal.pkl',
    './AP/2/vocal.pkl',
    './AP/3/vocal.pkl',
    './AP/3/vocal.pkl',
    './AP/4/vocal.pkl',
    './AP/4/vocal.pkl',]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_ret = pd.DataFrame()
    for i,model_path in enumerate(model_paths):
        print('BA model ', i, model_path)
        proba_list2 = []


        word_idx = pkl.load(open(vocab_list[i], 'rb'))
        df_process = hla2seq(df)
        test_loader = HLAData_test_pool(df_process,word_idx)

        model = torch.load(model_path).to(device)
        model.eval()

        with torch.no_grad():
            for batch_data in test_loader:
                if torch.cuda.is_available():
                    batch_data = batch_data.to(device)
                with torch.no_grad():
                    batch_data = Variable(batch_data)
                # print(batch_data)
                # torch.save(batch_data,f'batch_data_{i}.pth')
                out = model(batch_data)
                proba = F.softmax(out, dim=1)
                proba_list2.extend(proba.data[:, 1].cpu().numpy())
                
        df_ret[f'{i}'] = proba_list2
    print('now we are predict AP model, df is: \n',df)
    for i,model_path in enumerate(ap_model_paths):
        print('AP model ', i, model_path)
        proba_list2 = []
        model = torch.load(model_path).to(device)
        model.eval()
        word_idx = pkl.load(open(ap_vocab_list[i], 'rb'))
        test_loader = DataLoader_test(df, word_idx, device)
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data[0]
                if torch.cuda.is_available():
                    batch_data = batch_data.to(device)
                out = model(batch_data)
                proba = F.softmax(out, dim=1)
                proba_list2.extend(proba.data[:, 1].cpu().numpy())
                
        df_ret[f'{i+9}'] = proba_list2

    print(df_ret)
    df_ret['mean_proba_BA'] = df_ret[['0','1','2','3','4','5','6','7','8']].apply(lambda x: x.mean(),axis =1)
    df_ret['mean_proba_AP'] = df_ret[['9','10','11','12','13']].apply(lambda x: x.mean(),axis =1)
    df_ret['mean_proba'] = df_ret[['mean_proba_BA','mean_proba_AP']].apply(lambda x: x.mean(),axis =1)
    df_ret = df_ret.round(4)
    return df_ret['mean_proba_BA'].to_list(),df_ret['mean_proba_AP'].to_list(),df_ret['mean_proba'].to_list()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df = pd.read_csv('G:/R_code/ucla/pt_all.csv')
    df = pd.read_csv('./uploaded/21_AA_cut_new.csv')
    print(len(df))
    df['A1'] = df['A1'].map(lambda x: x.replace(':','').replace('*',''))
    df['A2'] = df['A2'].map(lambda x: x.replace(':','').replace('*',''))
    df['B1'] = df['B1'].map(lambda x: x.replace(':','').replace('*',''))
    df['B2'] = df['B2'].map(lambda x: x.replace(':','').replace('*',''))
    df['C1'] = df['C1'].map(lambda x: x.replace(':','').replace('*',''))
    df['C2'] = df['C2'].map(lambda x: x.replace(':','').replace('*',''))
    print('df: \n',df)
    # BA,AP,PS = ResMHApan_predict(df[['peptide','A1','A2','B1','B2','C1','C2']])
    model_paths = ['./BA/1/latest_best_0.9746_160.pth',
    './BA/1/latest_best_0.9748_169.pth',
    './BA/1/latest_best_0.9751_162.pth',
    './BA/2/latest_best_0.9747_160.pth',
    './BA/2/latest_best_0.9750_163.pth',
    './BA/2/latest_best_0.9753_161.pth',
    './BA/3/latest_best_0.9752_160.pth',
    './BA/3/latest_best_0.9755_163.pth',
    './BA/3/latest_best_0.9759_166.pth',
    ]
    ap_model_paths = ['./AP/1/latest_best_0.7912890967553345_123.pth',
    './AP/2/latest_best_0.7946202077250752_123.pth',
    './AP/3/latest_best_0.7935206869633099_103.pth',
    './AP/3/latest_best_0.7902829952416729_110.pth',
    './AP/4/latest_best_0.7911978985974129_107.pth',
    './AP/4/latest_best_0.7960330255508329_109.pth',]
    vocab_list = ['./BA/1/vocal.pkl',
    './BA/1/vocal.pkl',
    './BA/1/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/2/vocal.pkl',
    './BA/3/vocal.pkl',
    './BA/3/vocal.pkl',
    './BA/3/vocal.pkl',]
    ap_vocab_list = [    
        './AP/1/vocal.pkl',
    './AP/2/vocal.pkl',
    './AP/3/vocal.pkl',
    './AP/3/vocal.pkl',
    './AP/4/vocal.pkl',
    './AP/4/vocal.pkl',]
    df_ret = pd.DataFrame()
    for i,model_path in enumerate(model_paths):
        print('BA model ', i, model_path)
        proba_list2 = []


        word_idx = pkl.load(open(vocab_list[i], 'rb'))
        df_process = hla2seq(df)
        test_loader = HLAData_test_pool(df_process,word_idx)

        model = torch.load(model_path).to(device)
        model.eval()

        with torch.no_grad():
            for batch_data in test_loader:
                if torch.cuda.is_available():
                    batch_data = batch_data.to(device)
                with torch.no_grad():
                    batch_data = Variable(batch_data)
                # print(batch_data)
                # torch.save(batch_data,f'batch_data_{i}.pth')
                out = model(batch_data)
                proba = F.softmax(out, dim=1)
                proba_list2.extend(proba.data[:, 1].cpu().numpy())
                
        df_ret[f'{i}'] = proba_list2
    print('now we are predict AP model, df is: \n',df)
    for i,model_path in enumerate(ap_model_paths):
        print('AP model ', i, model_path)
        proba_list2 = []
        model = torch.load(model_path).to(device)
        model.eval()
        word_idx = pkl.load(open(ap_vocab_list[i], 'rb'))
        test_loader = DataLoader_test(df, word_idx, device)
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data[0]
                if torch.cuda.is_available():
                    batch_data = batch_data.to(device)
                out = model(batch_data)
                proba = F.softmax(out, dim=1)
                proba_list2.extend(proba.data[:, 1].cpu().numpy())
                
        df_ret[f'{i+9}'] = proba_list2

    print(df_ret)
    df_ret['mean_proba_BA'] = df_ret[['0','1','2','3','4','5','6','7','8']].apply(lambda x: x.mean(),axis =1)
    df_ret['mean_proba_AP'] = df_ret[['9','10','11','12','13','14']].apply(lambda x: x.mean(),axis =1)
    df_ret['mean_proba'] = df_ret[['mean_proba_BA','mean_proba_AP']].apply(lambda x: x.mean(),axis =1)
    df_ret = df_ret.round(4)
    #return df_ret['mean_proba_BA'].to_list(),df_ret['mean_proba_AP'].to_list(),df_ret['mean_proba'].to_list()
    df['BA'] = df_ret['mean_proba_BA'].to_list()
    df['AP'] = df_ret['mean_proba_AP'].to_list()
    df['PS'] = df_ret['mean_proba'].to_list()
    print('final result: \n',df)
    # if is_save:
    df.to_csv('./app/download/res_example.csv',index=False)
    # return df