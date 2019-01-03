# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = '../dataset/data/task2/tok/train.lc.norm.tok.{}.de'
    target_train = '../dataset/data/task2/tok/train.lc.norm.tok.{}.en'
    source_test = '../dataset/data/task2/tok/test_2016.lc.norm.tok.{}.de'
    target_test = '../dataset/data/task2/tok/test_2016.lc.norm.tok.{}.en'
    source_val = '../dataset/data/task2/tok/val.lc.norm.tok.{}.de'
    target_val = '../dataset/data/task2/tok/val.lc.norm.tok.{}.en'
    task1_de2016 = '../dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de'
    task1_en2016 = '../dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.en'
    task1_de2017 = '../dataset/data/task1/tok/test_2017_flickr.lc.norm.tok.de'
    task1_en2017 = '../dataset/data/task1/tok/test_2017_flickr.lc.norm.tok.en'
    # source_train = '../multi30k_bpe/task2_bpe/train.lc.norm.tok.{}.de.bpe'
    # target_train = '../multi30k_bpe/task2_bpe/train.lc.norm.tok.{}.en.bpe'
    # source_test = '../multi30k_bpe/task2_bpe/test_2016.lc.norm.tok.{}.de.bpe'
    # target_test = '../multi30k_bpe/task2_bpe/test_2016.lc.norm.tok.{}.en.bpe'
    # source_val = '../multi30k_bpe/task2_bpe/val.lc.norm.tok.{}.de.bpe'
    # target_val = '../multi30k_bpe/task2_bpe/val.lc.norm.tok.{}.en.bpe'
    #
    # test2016_en = '../multi30k_bpe/task1_bpe/test_2016_flickr.lc.norm.tok.en.bpe'
    # test2016_de = '../multi30k_bpe/task1_bpe/test_2016_flickr.lc.norm.tok.de.bpe'
    # test2017_en = '../multi30k_bpe/task1_bpe/test_2017_flickr.lc.norm.tok.en.bpe'
    # test2017_de = '../multi30k_bpe/task1_bpe/test_2016_flickr.lc.norm.tok.de.bpe'
    # training
    batch_size = 256 # alias = N
    batch_size_test = 64
    logdir_de2en = '../../logdir/logdir_de2en_1'
    logdir_en2de = '../../logdir/logdir_en2de_bpe'
    # model
    output_file = "out.txt"
    maxlen = 50 # Maximum number of words in a sentence.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 256
    lr = pow(hidden_units,-0.5) # learning rate. Using the same policy as the paper.
    warmup_step = 10000
    num_blocks = 2 # number of encoder/decoder blocks
    num_epochs = 10
    num_heads = 8
    dropout_rate = 0.2
    sinusoid = True # If True, use sinusoid. If false, positional embedding.

    
    
    
    
