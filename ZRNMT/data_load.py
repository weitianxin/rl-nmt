from __future__ import print_function
from hyperparams_rl import Hyperparams as hp
import numpy as np
import codecs

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



def create_rl_data(de_sents, en_sents, language="de", total=True):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    images = np.load(image_path.format("train")).shape[0]
    if total!=True:
        image_index = list(range(images))
    else:
        image_index = list(range(images))*5
    num = images
    x_list, image_index_list,y_list,Targets= [], [],[],[]
    for i,(de_sent,en_sent) in enumerate(zip(de_sents, en_sents)):
        if language=="de":
            x = [de2idx.get(word, 1) for word in (de_sent).split()]
            y = [en2idx.get(word, 1) for word in ("<S> "+en_sent+" </S>").split()]
        elif language=="en":
            x = [en2idx.get(word, 1) for word in (en_sent).split()]
            y = [de2idx.get(word, 1) for word in ("<S> "+de_sent+" </S>").split()]
        if max(len(x),len(y)) <=hp.maxlen:
            image_index_list.append(image_index[i])
            x_list.append(np.array(x))
            target_n = []
            for k in range(5):
                if language=="de":
                    target_n.append(en_sents[k * num + i % num])
                elif language=="en":
                    target_n.append(de_sents[k * num + i % num])
            Targets.append(target_n)
            y_list.append(np.array(y))
    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x,y) in enumerate(zip(x_list,y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))
    return X,np.array(image_index_list),Y,Targets

def load_rl_data(language,total=True):
    def _refine(line):
        return line.strip()
    de_sents, en_sents = [], []
    for i in range(1,6):
        if(total!=True):
            de_sents.extend([line.strip() for line in
                             codecs.open(hp.source_train.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in
                             codecs.open(hp.target_train.format(i), 'r', 'utf-8').readlines()])
        else:
            de_sents.extend([line.strip() for line in
                             codecs.open(hp.source_train.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in
                             codecs.open(hp.target_train.format(i), 'r', 'utf-8').readlines()])
    X,image_index,Y,Targets = create_rl_data(de_sents, en_sents,language,total)
    data = list(zip(X,image_index,Y,Targets))
    np.random.shuffle(data)
    X = [i for i,j,k,m in data]
    image_index = [j for i,j,k,m in data]
    Y = [k for i, j, k,m in data]
    Targets = [m for i, j, k,m in data]
    return X,image_index,Y,Targets

def create_test_rl_data(de_sents, en_sents, task, set="val", language="de"):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    num = len(en_sents) // 5
    if set=="val":
        total = 1014
    else:
        total = 1000
    # Index
    x_list,Targets,idents=  [],[],[]
    for i, (de_sent, en_sent) in enumerate(zip(de_sents, en_sents)):
        if language=="de":
            x = [de2idx.get(word, 1) for word in (de_sent).split()]  # 1: OOV, </S>: End of Text，忘了
            y = [en2idx.get(word, 1) for word in (u"<S> " + en_sent).split()]
        else:
            y = [de2idx.get(word, 1) for word in (u"<S> " + de_sent).split()]  # 1: OOV, </S>: End of Text，忘了
            x = [en2idx.get(word, 1) for word in (en_sent).split()]
        if max(len(x), len(y)) <= hp.maxlen:
            idents.append(i%total)
            x_list.append(np.array(x))
            if task=="task1":
                if language=="de":
                    Targets.append(en_sent)
                elif language=="en":
                    Targets.append(de_sent)
            else:
                target_n = []
                for k in range(5):
                    if language == "de":
                        target_n.append(en_sents[k * num + i % num])
                    elif language == "en":
                        target_n.append(de_sents[k * num + i % num])
                Targets.append(target_n)
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    for i, x in enumerate(x_list):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
    return  X,Targets,idents

def load_test_rl_data(set="val",language="de",task="task2"):
    de_sents, en_sents =[],[]
    if task=="task1":
        if set=="test2016":
            de_sents = [line.strip() for line in codecs.open(hp.task1_de2016, 'r', 'utf-8').readlines()]
            en_sents = [line.strip() for line in codecs.open(hp.task1_en2016, 'r', 'utf-8').readlines()]
        else:
            de_sents = [line.strip() for line in codecs.open(hp.task1_de2017, 'r', 'utf-8').readlines()]
            en_sents = [line.strip() for line in codecs.open(hp.task1_en2017, 'r', 'utf-8').readlines()]
    elif task=="task2":
        for i in range(1, 6):
            if set=="val":
                de_sents.extend([line.strip() for line in codecs.open(eval("hp.source_{}".format(set)).format(i), 'r', 'utf-8').readlines()])
                en_sents.extend([line.strip() for line in codecs.open(eval("hp.target_{}".format(set)).format(i), 'r', 'utf-8').readlines()])
            elif set=="test":
                de_sents.extend([line.strip() for line in
                                 codecs.open(eval("hp.source_{}".format(set)).format(i), 'r', 'utf-8').readlines()])
                en_sents.extend([line.strip() for line in
                                 codecs.open(eval("hp.target_{}".format(set)).format(i), 'r', 'utf-8').readlines()])

    X, Targets,idents = create_test_rl_data(de_sents, en_sents,task,set,language=language)
    data = list(zip(X,Targets,idents))
    np.random.seed(10)
    np.random.shuffle(data)
    X, Targets,idents = list(zip(*data))
    return X,Targets,idents




