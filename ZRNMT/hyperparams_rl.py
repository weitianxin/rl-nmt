class Hyperparams:
    '''Hyperparameters'''
    # data
    # source_train = '../dataset/data/task2/tok/train.lc.norm.tok.{}.de'
    # target_train = '../dataset/data/task2/tok/train.lc.norm.tok.{}.en'
    # source_test = '../dataset/data/task2/tok/test_2016.lc.norm.tok.{}.de'
    # target_test = '../dataset/data/task2/tok/test_2016.lc.norm.tok.{}.en'
    # source_val = '../dataset/data/task2/tok/val.lc.norm.tok.{}.de'
    # target_val = '../dataset/data/task2/tok/val.lc.norm.tok.{}.en'
    #
    source_train = '../multi30k_bpe/task2_bpe/train.lc.norm.tok.{}.de.bpe'
    target_train = '../multi30k_bpe/task2_bpe/train.lc.norm.tok.{}.en.bpe'
    source_test = '../multi30k_bpe/task2_bpe/test_2016.lc.norm.tok.{}.de.bpe'
    target_test = '../multi30k_bpe/task2_bpe/test_2016.lc.norm.tok.{}.en.bpe'
    source_val = '../multi30k_bpe/task2_bpe/val.lc.norm.tok.{}.de.bpe'
    target_val = '../multi30k_bpe/task2_bpe/val.lc.norm.tok.{}.en.bpe'
    task1_de2016 = '../multi30k_bpe/task1_bpe/test_2016_flickr.lc.norm.tok.de.bpe'
    task1_en2016 = '../multi30k_bpe/task1_bpe/test_2016_flickr.lc.norm.tok.en.bpe'
    task1_de2017 = '../multi30k_bpe/task1_bpe/test_2017_flickr.lc.norm.tok.de.bpe'
    task1_en2017 = '../multi30k_bpe/task1_bpe/test_2017_flickr.lc.norm.tok.en.bpe'

    logdir_cap_en = "../../logdir/logdir_cap_en_bpe"
    logdir_cap_de = "../../logdir/logdir_cap_de_bpe"
    # training
    batch_size = 64# 记得改过这个！
    batch_size_test = 64
    # model transformer
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 256 # alias = C
    #lr = pow(hidden_units,-0.5)#!!!0.00001
    lr = 0.00001
    # lr = 0.1
    warmup_step = 4000
    num_blocks = 2 # number of encoder/decoder blocks
    num_epochs = 2#记得改
    num_heads = 4
    dropout_rate_tran = 0.3
    sinusoid = True # If True, use sinusoid. If false, positional embedding.
    #lstm
    dropout_rate = 0.5
    lstm_units=512
    lstm_initializel_scale = 0.08
    lstm_drop_rate = 0.3
    attention_dim = 512
    decode_layerunit = 512
    lr_cap = 0.001
    hidden_units_cap = 512
    attention_loss_factor = 1.0
    clip_gradients = 5.0
    fc_kernel_initializer_scale = 0.08
    fc_kernel_regularizer_scale = 1e-4

    # de_train_cap = '../data/de_train_cap.txt'
    # en_train_cap = '../data/train_cap.txt'
    # en_test_cap = "../data/test_cap.txt"
    # de_test_cap = "../data/de_test_cap.txt"
    # de_val_cap = '../data/de_val_cap.txt'
    # en_val_cap = '../data/val_cap.txt'
    # logdir_cap_de = "logdir_cap_de"
    # logdir_cap_en = "logdir_cap_en"
    # en_train_single = '../data/train.1'

    
    
    
