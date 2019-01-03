from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

image_path = "../../image/flickr30k_ResNets50_blck4_{}.fp16.npy"

#dataset
def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('../preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

#hi
def create_data(source_sents, target_sents, set="special",mode="de2en"):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    source2idx,target2idx = de2idx,en2idx
    # Index
    num = len(target_sents) // 5
    x_list, y_list, y_target_list, Sources, Targets = [], [], [], [], []
    if mode == "en2de":
        source_sents,target_sents = target_sents,source_sents
        source2idx,target2idx = target2idx,source2idx
    if set == "special":
        for i, (source_sent, target_sent) in enumerate(zip(source_sents, target_sents)):
            x = [source2idx.get(word, 1) for word in (source_sent).split()]  # 1: OOV, </S>: End of Text，忘了
            y_target = [target2idx.get(word, 1) for word in (target_sent + u" </S>").split()]  # + u" </S>"
            y = [target2idx.get(word, 1) for word in (u"<S> " + target_sent).split()]
            if max(len(x), len(y)) <= hp.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                y_target_list.append(np.array(y_target))
                Sources.append(source_sent)
                target_n = []
                for k in range(5):
                    target_n.append(target_sents[k*num + i%num])
                Targets.append(target_n)
    else:
        for i, (source_sent, target_sent) in enumerate(zip(source_sents, target_sents)):
            x = [source2idx.get(word, 1) for word in (source_sent).split()]  # 1: OOV, </S>: End of Text，忘了
            y_target = [target2idx.get(word, 1) for word in (target_sent + u" </S>").split()]  # + u" </S>"
            y = [target2idx.get(word, 1) for word in (u"<S> " + target_sent).split()]
            if max(len(x), len(y)) <= hp.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                y_target_list.append(np.array(y_target))
                Sources.append(source_sent)
                Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    Y_target = np.zeros([len(y_target_list), hp.maxlen], np.int32)
    for i, (x, y, y_target) in enumerate(zip(x_list, y_list, y_target_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))
        Y_target[i] = np.lib.pad(y_target, [0, hp.maxlen - len(y_target)], 'constant', constant_values=(0, 0))

    return X, Y, Y_target, Sources, Targets


def load_train_data(mode="de2en"):
    de_sents,en_sents = [],[]
    for i in range(1,6):
        for j in range(1,6):
            de_sents.extend([line.strip() for line in
                        open(hp.source_train.format(i), mode='r', encoding='utf-8').readlines()])
            en_sents.extend([line.strip() for line in
                        codecs.open(hp.target_train.format(j), 'r', 'utf-8').readlines()])
    data = list(zip(de_sents, en_sents))
    np.random.shuffle(data)
    de_sents = [i for i, j in data]
    en_sents = [j for i, j in data]
    X, Y, Y_target, Sources, Targets = create_data(de_sents, en_sents, set="train",mode=mode)
    return X, Y, Y_target


def load_val_data(mode="de2en",task="task2"):
    de_sents, en_sents = [], []
    if task=="task2":
        for i in range(1, 6):
            de_sents.extend([line.strip() for line in
                             codecs.open(hp.source_val.format(i), 'r', 'utf-8').readlines()])
            en_sents.extend([line.strip() for line in
                             codecs.open(hp.target_val.format(i), 'r', 'utf-8').readlines()])
        set = "special"
    else:
        de_sents.extend([line.strip() for line in
                         codecs.open(hp.task1_de2016, 'r', 'utf-8').readlines()])
        en_sents.extend([line.strip() for line in
                         codecs.open(hp.task1_en2016, 'r', 'utf-8').readlines()])
        set = "normal"
    X, Y, Y_target, Sources, Targets = create_data(de_sents, en_sents, set=set,mode=mode)
    return X, Targets
