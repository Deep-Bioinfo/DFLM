def train_with_pre_training(train_l,test_l,ttf,encoder_name = 'human_3m1s_enc2-286'):
    path = Path('../human_data')
    res_score_list = []
    # for k in motif_family_dict:
        # try:
    if True:    
            # motif_family_path_l = motif_family_dict[k]
            # train_l = motif_family_path_l[::2]
            # test_l = motif_family_path_l[1::2]


            for i in range(len(train_l)):
                    print('motif:',train_l[i])
                    train_df = pd.read_csv(motif_path/train_l[i],sep='\t',compression='gzip', error_bad_lines=False)
                    train_df.seq = train_df.seq.map(clear_map)
                    train_df.dropna(axis=0,how='any',inplace=True)
                    if len(train_df) < 1000:
                        print('too smalll',len(train_df))
                        continue
                    # print(train_df.head())
                    len_val = int(len(train_df)*0.3)
                    train_df = train_df[len_val:]
                    valid_df = train_df[0:len_val]
                    # print(train_df.head())
                    test_df = pd.read_csv(motif_path/test_l[i],sep='\t',compression='gzip', error_bad_lines=False)
                    test_df.seq = test_df.seq.map(clear_map)
                    test_df.dropna(axis=0,how='any',inplace=True)

                    len_seq = 101#len(train_df.loc[0]['seq'])
                    train_df.drop(['FoldID','EventID'], axis = 1, inplace = True)
                    valid_df.drop(['FoldID','EventID'], axis = 1, inplace = True)
                    test_df.drop(['FoldID','EventID'], axis = 1, inplace = True)

                    df_neg = gen_negetive_by_shuffle(len_seq,train_df)
                    train_df = train_df.append(df_neg, ignore_index=True)
                    df_neg = gen_negetive_by_shuffle(len_seq,valid_df)
                    valid_df = valid_df.append(df_neg, ignore_index=True)
                    df_neg = gen_negetive_by_shuffle(len_seq,test_df)
                    test_df = test_df.append(df_neg, ignore_index=True)

                    voc = np.load(path/'human_vocab_3m1s.npy')
                    model_vocab = GenomicVocab(voc)
                    tok = Tokenizer(partial(GenomicTokenizer, ngram=3, stride=1), n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])
                    data_clas = GenomicTextClasDataBunch.from_df(path, train_df, valid_df, tokenizer=tok, vocab=model_vocab, min_freq = 100,
                                                                text_cols='seq', label_cols='Bound', bs=400)
                    clas_config = dict(emb_sz=200, n_hid=800, n_layers=6, pad_token=0, qrnn=False, output_p=0.4, 
                                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
                    drop_mult = 0.25
                    learn = get_model_clas(data_clas, drop_mult, clas_config)
                    #learn = get_model_clas(data_clas, drop_mult, clas_config)
                    
                    
                    learn.load_encoder(encoder_name)
                    learn = learn.to_fp16(dynamic=True);
                    learn.freeze()
                    try:
                        learn.lr_find()
                    except:
                        print('continue',train_l[i])
                        continue
                    # learn.recorder.plot()
                    learn.fit_one_cycle(3, 2e-2, moms=(0.8,0.7))
                    learn.freeze_to(-2)
                    learn.fit_one_cycle(3, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
                    learn.freeze_to(-3)
                    learn.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
                    learn.unfreeze()
                    learn.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
                    learn.fit_one_cycle(3, slice(1e-4/(2.6**4),1e-4), moms=(0.8,0.7))
                    ft_name = 'human_motif_deepram_all_classification2_3m1s286'
                    learn.save(ft_name)

                    learn.data = data_clas
                    res_score = get_scores(learn)
                    res_score.append(train_l[i])
                    res_score_list.append(res_score)
            write_list(res_score_list,'tf_family_3mtemp_classification.txt')
            return res_score_list




def train_with_fine_tune(motif_family_dict,tff):
    trained_key = []
    res_score_list = []
    count_k = 0
    print(motif_family_dict.keys())
    for k in motif_family_dict:
        try:
        # if True:    
            count_k = count_k + 1
            if k in trained_key or k=='CTCF': #or count_k < 21:
                continue
            print('kkkkkkkkkkkkkkkkkkkkkkk', k)
            brothers =  tff.find_brothers([k])
            if len(brothers) == 0:
                brothers.append([k])
            print('brother',brothers)
            motif_family_path_l = motif_family_dict[k]
            train_l = motif_family_path_l[::2]
            test_l = motif_family_path_l[1::2]
            train_l = motif_family_path_l[::2]
            print(train_l)
            Fine_Tune_df = pd.read_csv(motif_path/train_l[0],sep='\t',compression='gzip', error_bad_lines=False)
            for brother_k in brothers[0]:
                # print(brother_k,type(brother_k))
                if not(brother_k in motif_family_dict.keys()):
                    continue
                print(brother_k)
                trained_key.append(brother_k)
                motif_family_path_l = motif_family_dict[brother_k]

                if k != brother_k:
                    print(brother_k,type(brother_k))
                    train_l.extend(motif_family_path_l[::2])
                    test_l.extend(motif_family_path_l[1::2])
                    Fine_Tune_df = pd.read_csv(motif_path/train_l[0],sep='\t',compression='gzip', error_bad_lines=False)

                for motif_p in train_l[1:-1]:
                    tm_df = pd.read_csv(motif_path/motif_p,sep='\t',compression='gzip', error_bad_lines=False)
                    Fine_Tune_df = Fine_Tune_df.append(tm_df)

                Fine_Tune_df.drop(['FoldID','EventID'], axis = 1, inplace = True)
                len_df = len(Fine_Tune_df)
                Fine_Tune_df = Fine_Tune_df[0:int(len_df*0.2)]
                print('brothers',len(Fine_Tune_df), len(train_l))
            rorder = np.random.permutation(len(Fine_Tune_df))
            Fine_Tune_df = Fine_Tune_df.take(rorder)
            len_val = int(len(Fine_Tune_df)*0.3)
            tf_train_df = Fine_Tune_df[len_val:]
            tf_valid_df = Fine_Tune_df[0:len_val]



            tf_train_df.seq = tf_train_df.seq.map(clear_map)
            tf_train_df.dropna(axis=0,how='any',inplace=True)

            tf_valid_df.seq = tf_valid_df.seq.map(clear_map)
            tf_valid_df.dropna(axis=0,how='any',inplace=True)

            voc = np.load(path/'human_vocab_3m1s.npy')
            model_vocab = GenomicVocab(voc)
            len_seq = 101 #len(tf_valid_df.loc[0]['seq'])

            df_neg = gen_negetive_by_shuffle(len_seq,tf_train_df)
            tf_train_df = tf_train_df.append(df_neg, ignore_index=True)

            df_neg = gen_negetive_by_shuffle(len_seq,tf_valid_df)
            tf_valid_df = tf_valid_df.append(df_neg, ignore_index=True)

            tok = Tokenizer(partial(GenomicTokenizer, ngram=3, stride=1), n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])
            data = GenomicTextLMDataBunch.from_df(path, tf_train_df, tf_valid_df, bs=800, tokenizer=tok, 
                                              chunksize=10000, text_cols='seq', label_cols='Bound', vocab=model_vocab, min_freq=100)
            config = dict(emb_sz=200, n_hid=800, n_layers=6, pad_token=0, qrnn=False, output_p=0.25, 
                                          hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15, tie_weights=True, out_bias=True)
            drop_mult = 0.5
            learn = get_model_LM(data, drop_mult, config)

            learn = learn.to_fp16(dynamic=True);
            learn.load('human_3m1s2h286');
            learn.lr_find()
            # learn.recorder.plot()
            learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))
            brothers_name = ''
            for i in range(len(brothers[0])):
                brothers_name = brothers_name + brothers[0][i]
            tf_family_learn_name = 'human_motif_deepbind2_3m1s-286'
            tf_family_encoder_name = 'human_motif_deepbind2_3m1s_enc-286'
            learn.save(tf_family_learn_name)
            learn.save_encoder(tf_family_encoder_name)
            
            '''
            a = tfs_df[0].map(lambda x : x+'_AC.seq.gz')
            b = tfs_df[0].map(lambda x : x+'_B.seq.gz')
            train_tag_l = list(set(a).intersection(set(train_l)))
            test_tag_l = list(set(b).intersection(set(test_l)))
            print(train_tag_l)
            print(test_tag_l)
            '''
            tf_score = train_with_pre_training(train_l,test_l,tff,tf_family_encoder_name)
            res_score_list.extend(tf_score)
        except:
            print('some error')
            continue

    return res_score_list


#!/usr/bin/env python
# coding: utf-8



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx
from random import choice

sys.path.append("../../..")
from gene_utils import *
from TFFamily import *

motif_path = Path('/home/sc2/Documents/deepbind-motif-data/encode-deepbind')
path = Path('../human_data')

tfs_df = pd.read_csv('~/Documents/deepbind-motif-data/TF_CellLine.txt',header=None)

# motif_f_list_path = 'TF_CellLine.txt'
# tf_family_name = 'ATF2'


# # LM Fine Tune

# Here we create a language model corpus from our classification dataset. This basically concatenates all our promoter sequences into a single long string of text. The language model is trained on the concatenated promoter corpus. We need to make sure to create our dataloader with the correct vocabulary.


def motif_family_init(motif_path):
    list_f = os.listdir(motif_path)
    list_f = sorted(list_f)
    ptf = list_f[0].split('_')[0]
    motif_family_dict = {}
    motif_family_list = []
    for i in range(len(list_f)):
        f_name = list_f[i]
        tf = f_name.split('_')[0]
        # print(f_name)
        if tf == ptf:
            # print(i)
            motif_family_list.append(f_name)
        else:
            motif_family_dict[ptf] = motif_family_list
            motif_family_list = []
            motif_family_list.append(f_name)
            # print(i)
        ptf = f_name.split('_')[0]
    return motif_family_dict


def clear_map(x):
    if (x.find('N')!=-1) or (x.find('n')!=-1):
        print(x)
        x = None

    return x

def gen_negetive_by_random(len_neg, len_neg_df):
    data = {}
    random_seq = []
    for j in range(len_neg_df):
              
        random_word = ''
        for i in range(len_neg):
            random_w = choice(['A','T','C','G'])
            random_word += random_w
        random_seq.append(random_word)
    data = ({'seq':Series(random_seq)})
    
    df = DataFrame(data)
    df.insert(df.shape[1],'Bound',0)
    return df

def gen_negetive_by_shuffle(len_neg, target_df):
    data = {}
    random_seq = []
    # print(len(target_df))
    for index, row in target_df.iterrows():
        target_seq = row['seq']

        random_word = dinucshuffle(target_seq)
        random_seq.append(random_word)
    data = ({'seq':Series(random_seq)})
    
    df = DataFrame(data)
    df.insert(df.shape[1],'Bound',0)
    return df

def dinucshuffle(sequence):
    b=[sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d

def write_list(res_socre_list,res_path='res_deepbind_all_score_temp.txt'):
    wf = open(res_path,'w')
    for i in range(len(res_socre_list)):
        for j in range(len(res_socre_list[i])):
            wf.write(str(res_socre_list[i][j]))
            wf.write(',')
        wf.write('\n')
    wf.close()

res_score_list = []
def main():
    
    
    motif_family_dict = {}
    motif_family_dict = motif_family_init(motif_path)
    tff = TFFamily('../../../tree')
    # tf = tfs_df[0].map(lambda x : x.split('_')[0])
    # tf_k = sorted(list(set(tf)))[17:]
    # print(tf_k)
    test_dict = {}
    # print(motif_family_dict['BATF'])
    # for k in tf_k:
    #     test_dict[k] = motif_family_dict[k]
    # test_dict[tf_name]= motif_family_dict[tf_name]
    res_score_list = train_with_fine_tune(motif_family_dict, tff)
    # res_score_list = train_with_fine_tune(test_dict, tff)
    print(res_score_list)
    write_list(res_score_list, 'deepram_'+'3mall'+'.txt')


main()

