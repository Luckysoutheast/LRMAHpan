# utils.py

import torch
import torch.nn.functional as F
from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
import torch.utils.data as Data

def roberta_base_AdamW_LLRD(model):

    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = 3.5e-6 
    head_lr = 3.6e-6
    lr = init_lr


    # === 12 Hidden layers ==========================================================

    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       

        lr *= 0.9     

    # === Embeddings layer ==========================================================

    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        

    return opt_parameters, init_lr


def prepare_pack_padded_sequence( inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        # self.TEXT = None
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        # NLP = spacy.load('en')
        # tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        tokenizer = lambda sent: [x for x in sent if x != " "]

        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False, batch_first=True, include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        #strs = ' YATLEVWDRNSIKGH,QFMPC'
        #strs = ' YATLEWRVNSDIHKGQ,FMPC'
        strs = ' YATLEWVNRDSIGKHF,QMPC'
        pep_list = []
        for i in range(1,22):
            pep_list.append(strs[i]*(22-i))
        vocab_df = pd.DataFrame({'text':pep_list,'label':[1]*21})
        vocab_df_examples = [data.Example.fromlist(i, datafields) for i in vocab_df.values.tolist()]
        vocab_df_data = data.Dataset(vocab_df_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.95)
        
        TEXT.build_vocab(vocab_df_data)
        
        self.vocab = TEXT.vocab
        #np.save('../data/vocab_dict.npy', TEXT.vocab.stoi)
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))
        print ("vocab {}".format(self.vocab.itos))



def make_data(data, vocab_dict):
    pep_inputs, hla_inputs, labels = [], [], []
    pep_lens = []
    for pep in data.text:
        pep_lens.append(len(pep))
        tokenizer = lambda sent: [x for x in sent if x != " "]
        tokenized = tokenizer(pep)
        #if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (50 - len(tokenized))
        pep_input = [[vocab_dict[t] for t in tokenized]]
        # pep = pep.ljust(50, '')
        # pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        # hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        # hla_inputs.extend(hla_input)
        # labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(pep_lens)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, pep_lens):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        # self.hla_inputs = hla_inputs
        # self.labels = labels
        self.pep_lens = pep_lens

    def __len__(self): # 样本数
        return self.pep_inputs.shape[0] # 改成hla_inputs也可以哦！

    def __getitem__(self, idx):
#         return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx],self.pep_lens[idx]
#         print(self.hla_inputs[idx])
#         print(self.pep_inputs[idx])
        return self.pep_inputs[idx], self.pep_lens[idx]


def predict_dataloader(test_df,vocab):

    pep_inputs, pep_lens = make_data(test_df, vocab)
    data_loader = Data.DataLoader(MyDataSet(pep_inputs, pep_lens), 512, shuffle = False, num_workers = 0)
    return test_df, pep_inputs, pep_lens, data_loader


def load_data_predict(test_df,strs):
    tokenizer = lambda sent: [x for x in sent if x != " "]
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False, batch_first=True, include_lengths=True)#, fix_length=50
    datafields = [("text",TEXT)]
    test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
    test_data = data.Dataset(test_examples, datafields)

    # strs = ' YATLEWVNRDSIGKHF,QMPC'
    pep_list = []
    for i in range(1,22):
        pep_list.append(strs[i]*(22-i))
    vocab_df = pd.DataFrame({'text':pep_list})
    vocab_df_examples = [data.Example.fromlist(i, datafields) for i in vocab_df.values.tolist()]
    vocab_df_data = data.Dataset(vocab_df_examples, datafields)
    TEXT.build_vocab(vocab_df_data)
    vocab_str = TEXT.vocab.stoi
    print('compare vocab:\n',vocab_str)
    test_iterator = data.Iterator(
        test_data,
        batch_size=256,
        train=False,
        # sort_key=lambda x: len(x.text),
        sort=False,
        repeat=False,
        shuffle=False)
    return test_iterator, vocab_str




def seq_len_to_mask(seq_len, max_len=None): #50
    r"""
    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.
    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])
    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

from torch import nn as nn
def get_embeddings(init_embed, padding_idx=None):
    r"""
    根据输入的init_embed返回Embedding对象。如果输入是tuple, 则随机初始化一个nn.Embedding; 如果输入是numpy.ndarray, 则按照ndarray
    的值将nn.Embedding初始化; 如果输入是torch.Tensor, 则按该值初始化nn.Embedding; 如果输入是fastNLP中的embedding将不做处理
    返回原对象。
    :param init_embed: 可以是 tuple:(num_embedings, embedding_dim), 即embedding的大小和每个词的维度;也可以传入
        nn.Embedding 对象, 此时就以传入的对象作为embedding; 传入np.ndarray也行，将使用传入的ndarray作为作为Embedding初始化;
        传入torch.Tensor, 将使用传入的值作为Embedding初始化。
    :param padding_idx: 当传入tuple时，padding_idx有效
    :return nn.Embedding:  embeddings
    """
    if isinstance(init_embed, tuple):
        res = nn.Embedding(
            num_embeddings=init_embed[0], embedding_dim=init_embed[1], padding_idx=padding_idx)
        nn.init.uniform_(res.weight.data, a=-np.sqrt(3 / res.weight.data.size(1)),
                         b=np.sqrt(3 / res.weight.data.size(1)))
    elif isinstance(init_embed, nn.Module):
        res = init_embed
    elif isinstance(init_embed, torch.Tensor):
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    elif isinstance(init_embed, np.ndarray):
        init_embed = torch.tensor(init_embed, dtype=torch.float32)
        res = nn.Embedding.from_pretrained(init_embed, freeze=False)
    else:
        raise TypeError(
            'invalid init_embed type: {}'.format((type(init_embed))))
    return res


'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def evaluate_model(model, iterator):
    model.eval()
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text[0].cuda()
            l = batch.text[1].cuda()
        else:
            x = batch.text[0]
            l = batch.text[1]
        #x = x.permute(1, 0)
        y_pred = model(x, l)
        # loss = F.cross_entropy(y_pred['pred'].cpu().data, batch.label.numpy())
        predicted = torch.max(y_pred['pred'].cpu().data, 1)[1] + 1
        # correct = torch.argmax(y_pred, 1) == (batch.label.numpy()-1)
        # log(model, loss.cpu(), correct.cpu())
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    #print("all_y shape {}".format(len(all_y)))
    #print("all_preds shape {}".format(len(all_preds)))
    print(classification_report(all_y, all_preds))
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

def predict(model, sentence, vocab_dict, device):
    model.eval()
    tokenizer = lambda sent: [x for x in sent if x != " "]
    tokenized = tokenizer(sentence)
    #if len(tokenized) < min_len:
    #    tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [vocab_dict[t] for t in tokenized]
    # print(indexed)
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    l = torch.tensor([len(tokenized)])
    # print(tensor)
    y_pred = model(tensor, l)
    # print(F.softmax(y_pred['pred'], dim=1))
    pro = F.softmax(y_pred['pred'], dim=1)
    predicted = torch.max(y_pred['pred'].cpu().data, 1)[1] + 1
    # print(predicted)
    return predicted.item(), pro.detach().numpy()[0]

# def predict_mul(model, df, vocab_dict, device, name):
#     '''
#         DataFrame: peptide  allele
#                 AAAAAAA  HLA-A0201
#     '''
#     df['allele'] = df['allele'].map(lambda x: x.replace(':',''))
#     df['allele'] = df['allele'].map(lambda x: x.replace('*',''))
#     allele = pd.read_csv('../data/class1_pseudosequences.csv')
#     df2 = df.merge(allele)
#     del df2['allele']
#     one_list = []
#     proba_list1 = []
#     proba_list2 = []
#     proba_list3 = []
#     proba_list4 = []
#     proba_list5 = []
#     proba_list6 = []
#     for index, row in df2.iterrows():
#         sentence = row['peptide'] + ',' + row['pseudosequence']
#         # print(sentence)
#         one, proba = predict(model, sentence, vocab_dict, device)
#         print(one)
#         one_list.append(one)
#         proba_list1.append(proba[0])
#         proba_list2.append(proba[1])
#         proba_list3.append(proba[2])
#         proba_list4.append(proba[3])
#         proba_list5.append(proba[4])
#         proba_list6.append(proba[5])
#     # print(pro_list)
#     df['one'] = one_list
#     df['proba1'] = proba_list1
#     df['proba2'] = proba_list2
#     df['proba3'] = proba_list3
#     df['proba4'] = proba_list4
#     df['proba5'] = proba_list5
#     df['proba6'] = proba_list6
#     # print(df)
#     df.to_csv(f'./work/{name}.csv',index=False)
    #df.to_csv('./work/neo_tcell_2021_a0301_26_best3.csv',index=False)


# def hla2seq(hla):
#     allele = pd.read_csv('../data/class_a.csv')
#     return allele[allele['allele'] == hla]['']

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_dict = np.load('./work/vocab_dict_iedb_6label_best.npy', allow_pickle=True).item()
    #vocab_dict = np.load('../data/vocab_dict_iedb_6label_999.npy', allow_pickle=True).item()
    #vocab_dict = np.load('../data/vocab_dict_iedb_6label_999.npy', allow_pickle=True).item()
    print(vocab_dict)
    #model = torch.load('./work/latest_iedb_best_0.8872956332008249_9.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9051016767748841_22.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9001070281840885_16.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9015340706386015_20.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9032510771641207_35.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9012925969447708_26.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9130434782608695_80.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.9044261652957305_106.pth', map_location='cpu').to(device)
    model = torch.load('./work/latest_iedb_best_0.9063846455150802_101.pth', map_location='cpu').to(device)
    #model = torch.load('./work/latest_iedb_best_0.906528719229397_15.pth', map_location='cpu').to(device)
    #sentence = 'TLQELSHAL,YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY'
    #pred = predict(model, sentence, vocab_dict, device)
    #hlas = ['A0301','A1101','A2402','A2601','A3101','A3303','B4001','B5701','C0701']
    #hlas = ['A0301','A3001','A6801','B0702','B0801','B1402','B1501','B1801','B2705','B3501','B3901','B4402','B4403','B5801','C0802']
    #hlas = ['A0101']
    #hlas = ['A0101','A0201','A0203','A0206','A0207','A0301','A1101','A2402','A2601','A3001','A3101','A3303','A6801','B0702','B0801','B1402','B1501','B1801','B2705','B3501','B3901','B4001','B4402','B4403','B5101','B5701','B5801','C0701','C0802']
    #hlas = ['A0101','A0203','C0701','B4001','B5701']
    #hlas = ['HLA-A2608','HLA-B0706','SLA-11301','SLA-20101','SLA-20502','SLA-21001','SLA-21101','SLA-21201','SLA-30101','SLA-30701']
    #hlas=['A0201','A0203','A0206','A0207','A0301','A1101','A2402','A2601','A3001','A3101','A3303','A6801','B0702','B0801','B1402','B1501','B1801','B2705','B3501','B3901','B4001','B4402','B4403','B5101','B5701','B5801','C0701','C0802']
    hlas = ['HLA-A0205', 'HLA-A2301','HLA-A2501', 'HLA-A2608', 'HLA-A2902', 'HLA-A3201', 'HLA-A6601', 'HLA-A6802', 'HLA-A6901', 'HLA-B0706', 'HLA-B1302', 'HLA-B1502', 'HLA-B3503', 'HLA-B3508', 'HLA-B3701', 'HLA-B4002', 'HLA-B4101', 'HLA-B4501', 'HLA-B4901', 'HLA-B5001', 'HLA-C0102', 'HLA-C0202', 'HLA-C0303', 'HLA-C0304', 'HLA-C0401', 'HLA-C0501', 'HLA-C0602', 'HLA-C0702', 'HLA-C0704', 'HLA-C0706', 'HLA-C1203', 'HLA-C1402', 'HLA-C1505', 'HLA-C1601', 'HLA-C1701', 'SLA-10401', 'SLA-21201']

    for hla in hlas:
        print(hla)
    #hla = 'HLA-A0206'
        df = pd.read_csv(f'../data/2021_test/{hla}.csv')
    #for m in ['latest_iedb_best_0.9032510771641207_35.pth','latest_iedb_best_0.9012925969447708_26.pth','latest_iedb_best_0.8962005483744614_27.pth']:
    #    model = torch.load(f'./work/{m}', map_location='cpu').to(device)
    #    predict_mul(model, df, vocab_dict, device,'2021_'+hla+'_'+m.split('_')[-1].split('.')[0])
        predict_mul(model, df, vocab_dict, device,f'2021_{hla}_101_Acc')
    

    #print(model.enc.encoder.ring_att[2].attn[0,1][:,1].data.shape)
    
    #print(model.enc.encoder.ring_att[0].attn.data.shape)
    #import matplotlib.pyplot as plt
    #import seaborn
    #seaborn.set_context(context="talk")
    #def draw(data, x, y, ax):
    #    seaborn.heatmap(data,xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=True, ax=ax)

    #sent = 'F M N P H L I S V , Y F A M Y G E K V A H T H V D T L Y V R Y H Y Y T W A V L A Y T W Y'.split()
    #for layer in range(0, 3):
    #    fig, axs = plt.subplots(5,1, figsize=(20, 10))
    #    print("Encoder Layer", layer+1)
    #    for h in range(5):
    #        draw(model.enc.encoder.star_att[layer].attn[0,h].data,sent, [], ax=axs[h])
    #    plt.show()

