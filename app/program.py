from copy import deepcopy
import os
from unittest.mock import sentinel
import pandas as pd
import numpy as np
import collections
import re
import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_sequence
from utils import *
import random
import sys
import pickle as pkl
import copy
from torch.autograd import Variable
from predict import one_hla_st,six_hla_st 

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速
setup_seed(24)

def file_process(is_ma,upload="./uploaded/multiple_query.csv"):
    df = pd.read_csv(upload)
    print(df)
    print('================ma: ',is_ma, is_ma == 'True')
    if is_ma == True:
        print('ResMAHPan predicting......')
        if len(df.columns) < 7:
            df_candicated_all_hla = ResMHApan_batch(df['peptide'].to_list(), df['allele'][0],is_save)
        else:
            df_candicated_all_hla = ResMHApan_batch2(df)
    else:
        print('STMHCPan predicting......')
        if len(df.columns) < 3:
            print('len(df.columns) < 3')
            df_candicated_all_hla = one_hla_st(df)
        else:
            print('len(df.columns) >= 3')
            df_candicated_all_hla = six_hla_st(df)
    df_candicated_all_hla.to_csv('./app/download/result.csv',index=None)
    return df_candicated_all_hla


def check_peptide(peptides):
    if len(peptides) == 0: return False
    cond = True
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    for peptide in peptides.split():
        if len(peptide) < 7 or len(peptide) > 15:
            cond = False
            break
        elif not all(c in amino for c in peptide):
            cond = False
            break
    return cond

def check_mhc(mhcs):
    if len(mhcs) == 0: return False
    cond = True
    import re
    # HLA-A0201
    for mhc in mhcs.split():
        mhc = mhc.replace('*','').replace(':','')
        z = re.match(r"^HLA-[ABC]\d{4}$",mhc)
        if not z:
            cond = False
            break
    return cond


def check_upload(upload):
    import pandas as pd
    try:
        df = pd.read_csv(upload)
    except:
        print('hey1')
        return False
    else:
        try: assert df.shape[1] == 2
        except AssertionError: print('hey2');return False
        else: 
            try: 
                for i in range(df.shape[0]):
                    peptide = df.iloc[i][0]
                    mhc = df.iloc[i][1]
                    assert check_peptide(peptide) and check_mhc(mhc)
            except AssertionError: print(i,peptide,mhc);return False
            else: return True

# allele = pd.read_csv('../data/class1_pseudosequences.csv')

def slide_cut(peptide):
    #LVLPVVLQLKLFLRECKVANY
    candicated_pep = []
    for l in range(8,12):
        for start_idx in range(len(peptide)-l+1):
            # print(start_idx, start_idx+l)
            candicated_pep.append(peptide[start_idx:start_idx+l])
    return candicated_pep

def presentation_compute_sa(peptides, hlas, is_neoantigen=True, is_save=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peptide_list = peptides.split()
    df_candicated_all_hla = pd.DataFrame()
    for hla in hlas.split():
        print('processing ', hla)
        hla= hla.replace('*','').replace(':','')
        print('processing ', hla)
        df_candicated = pd.DataFrame({'peptide':peptide_list,'HLA':hla})
        print('raw dataFrame: \n',df_candicated)
        df_candicated = one_hla_st(df_candicated)
        df_candicated_all_hla = df_candicated_all_hla.append(df_candicated)
        
    if is_save:
        df_candicated_all_hla.to_csv('./app/download/result.csv',index=False)
    print('final result" \n',df_candicated_all_hla)
    return df_candicated_all_hla



#========MA================
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

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

import torch.utils.data as Data

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
    
def data_process(df, word_idx, max_len=59):
    # df_ = deepcopy(df)
    df_output = pd.DataFrame()
    for a in ['pep_a1','pep_a2','pep_b1','pep_b2','pep_c1','pep_c2']:
        df_output[a] = df[a].map(lambda x: [word_idx[word] for word in x[:max_len]])
        df_output[a] = df_output[a].map(lambda x: x + [0] * (max_len - len(x)))
    df_output['seq'] = df_output.apply(lambda x: np.array([x['pep_a1'],x['pep_a2'],x['pep_b1'],x['pep_b2'],x['pep_c1'],x['pep_c2']]), axis=1)
    return df_output[['seq']]
    
def hla2seq(df):
    df_cp = copy.deepcopy(df)
    print('df_cp \n',df_cp)
    hlaseq = pd.read_csv('allele_sequences.csv')
    hlaseq['allele'] = hlaseq['allele'].map(lambda x: x.replace(':',''))
    hlaseq['allele'] = hlaseq['allele'].map(lambda x: x.replace('*',''))
    for it in ['A1','A2','B1','B2','C1','C2']:
        df_cp=df_cp.merge(hlaseq,left_on=it, right_on='allele')
    df_cp = df_cp[['peptide','sequence_x','sequence_y']]
    # print('df2 ',df)
    df_cp.columns = ['peptide','A1','B1','C1','A2','B2','C2']
    df_cp['pep_a1'] = df_cp.apply(lambda x: x['peptide'] + x['A1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_a2'] = df_cp.apply(lambda x: x['peptide'] + x['A2'] + x['peptide'][::-1],axis=1)
    df_cp['pep_b1'] = df_cp.apply(lambda x: x['peptide'] + x['B1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_b2'] = df_cp.apply(lambda x: x['peptide'] + x['B2'] + x['peptide'][::-1],axis=1)
    df_cp['pep_c1'] = df_cp.apply(lambda x: x['peptide'] + x['C1'] + x['peptide'][::-1],axis=1)
    df_cp['pep_c2'] = df_cp.apply(lambda x: x['peptide'] + x['C2'] + x['peptide'][::-1],axis=1)
    df_cp = df_cp[['peptide', 'pep_a1', 'pep_a2', 'pep_b1', 'pep_b2', 'pep_c1', 'pep_c2']]
    return df_cp

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
    df_ret['mean_proba_AP'] = df_ret[['9','10','11','12','13','14']].apply(lambda x: x.mean(),axis =1)
    df_ret['mean_proba'] = df_ret[['mean_proba_BA','mean_proba_AP']].apply(lambda x: x.mean(),axis =1)
    df_ret = df_ret.round(4)
    return df_ret['mean_proba_BA'].to_list(),df_ret['mean_proba_AP'].to_list(),df_ret['mean_proba'].to_list()

def ResMHApan(peptide, a1, b1, c1, a2, b2, c2):
    df = pd.DataFrame({'peptide':peptide,'A1':a1,'B1':b1,'C1':c1, 'A2':a2, 'B2':b2, 'C2':c2}, index=[0])
    probability = ResMHApan_predict(df)
    flags = 'off'
    return probability, flags

def ResMHApan_df(df):
    BA,AP,PS = ResMHApan_predict(df)
    flags = 'off'
    return BA,AP,PS,flags

def ResMHApan_batch(peptides, mhcs, is_save):
    if isinstance(mhcs, str):
        a1,a2,b1,b2,c1,c2 = mhcs.split()
    elif isinstance(mhcs, list):
        assert len(mhcs) == 6
        a1,a2,b1,b2,c1,c2 = mhcs
    a1 = a1.replace(':','').replace('*','')
    a2 = a2.replace(':','').replace('*','')
    b1 = b1.replace(':','').replace('*','')
    b2 = b2.replace(':','').replace('*','')
    c1 = c1.replace(':','').replace('*','')
    c2 = c2.replace(':','').replace('*','')
    print('peptides: \n', peptides)
    print('mhcs: \n',a1,a2,b1,b2,c1,c2)
    
    if isinstance(peptides, str):
        df = pd.DataFrame({'peptide':peptides.split(),'A1':a1,'A2':a2,'B1':b1, 'B2':b2, 'C1':c1, 'C2':c2})
    elif isinstance(peptides, list):
        df = pd.DataFrame({'peptide':peptides,'A1':a1,'A2':a2,'B1':b1, 'B2':b2, 'C1':c1, 'C2':c2})
    print('df: \n',df)
    BA,AP,PS = ResMHApan_predict(df)
    df['BA'] = BA
    df['AP'] = AP
    df['PS'] = PS
    print('final result: \n',df)
    if is_save:
        df.to_csv('./app/download/result.csv',index=False)
    return df


def ResMHApan_batch2(df):
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
    df.to_csv('./app/download/result.csv',index=False)
    return df