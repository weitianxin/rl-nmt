from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
image_path = "../../image/flickr30k_ResNets50_blck4_{}.fp16.npy"
def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/de.vocab_bpe.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/en.vocab_bpe.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_test_cap_data(source_sents, target_sents,set,language="de"):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    images = np.load(image_path.format(set)).shape[0]
    image_index = list(range(images))
    # Index
    Targets,image_index_list = [], []
    for i in range(images):
        target_n = []
        if language=="de":
            for k in range(5):
                target_n.append(source_sents[i+images*k])
            flag = all([len(sen.split())<=hp.maxlen for sen in target_n])
        else:
            for k in range(5):
                target_n.append(target_sents[i+images*k])
            flag = all([len(sen.split())<=hp.maxlen for sen in target_n])
        if flag:
            image_index_list.append(image_index[i])
            Targets.append(target_n)
    return  np.array(image_index_list),Targets

def load_test_cap_data(set="val",language="de"):
    de_sents, en_sents = [], []
    for i in range(1,6):
        if set=="val":
            de_sents.extend([line.strip() for line in codecs.open(hp.source_val.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in codecs.open(hp.target_val.format(i), 'r', 'utf-8').readlines()])
        elif set=="train":
            de_sents.extend([line.strip() for line in codecs.open(hp.source_train.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in codecs.open(hp.target_train.format(i), 'r', 'utf-8').readlines()])
        elif set=="test":
            de_sents.extend([line.strip() for line in codecs.open(hp.source_test.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in codecs.open(hp.target_test.format(i), 'r', 'utf-8').readlines()])
    image_index,Targets = create_test_cap_data(de_sents, en_sents,set,language=language)
    return image_index,Targets

def create_cap_data(sents,set,total=True): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    images = np.load(image_path.format("train")).shape[0]
    if total!=True:
        image_index = list(range(images))
    else:
        image_index = list(range(images))*5
    x_list, Targets, image_index_list,x_target_list= [], [], [],[]
    for i,sent in enumerate(sents):
        if set=="de":
            x = [de2idx.get(word, 1) for word in (sent).split()] 
            x_target = [de2idx.get(word, 1) for word in (sent+u" </S>").split()] 
        else:
            x = [en2idx.get(word, 1) for word in (sent).split()] 
            x_target = [en2idx.get(word, 1) for word in (sent+u" </S>").split()]
        if max(len(x),len(x_target)) <=hp.maxlen:
            image_index_list.append(image_index[i])
            x_list.append(np.array(x))
            Targets.append(sent)
            x_target_list.append(np.array(x_target))
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    X_target = np.zeros([len(x_target_list), hp.maxlen], np.int32)
    for i, (x,x_target) in enumerate(zip(x_list,x_target_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        X_target[i] = np.lib.pad(x_target, [0, hp.maxlen-len(x_target)], 'constant', constant_values=(0, 0))
    return X,np.array(image_index_list), Targets,X_target

def load_cap_data(set,total=True):
    def _refine(line):
        return line.strip()
    sents = []
    for i in range(1,6):
        if set=="de":
            if(total!=True):
                sents.extend([_refine(line) for line in open(hp.source_train.format(i), 'r',encoding="utf-8").readlines()])
            else:
                sents.extend([_refine(line) for line in open(hp.source_train.format(i), 'r',encoding="utf-8").readlines()])
        else:
            if(total!=True):
                sents.extend([_refine(line) for line in open(hp.target_train.format(i), 'r').readlines()])
            else:
                sents.extend([_refine(line) for line in open(hp.target_train.format(i), 'r').readlines()])
    X,image_index,Targets,X_target = create_cap_data(sents,set,total)#!
    data = list(zip(X,image_index,Targets,X_target))
    np.random.shuffle(data)
    X = [i for i,j,k,m in data]
    image_index = [j for i,j,k,m in data]
    Targets = [k for i,j,k,m in data]
    X_target = [m for i,j,k,m in data]
    return X,image_index,Targets,X_target

