# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import regex
from collections import Counter
import time
def make_vocab(set, fname):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''
    text = []
    for i in range(1,6):
        text.extend(open("multi30k_bpe/task2_bpe/train.lc.norm.tok.{}.{}.bpe".format(i,set), 'r', encoding='utf-8').read().split())
    word2cnt = Counter(text)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with open('preprocessed/{}'.format(fname), mode='w', encoding='utf-8') as fout:
        #this is hz
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab("de", "de.vocab_bpe.tsv")
    make_vocab("en", "en.vocab_bpe.tsv")
    print("Done")
