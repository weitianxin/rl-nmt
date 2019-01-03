from __future__ import print_function
import tensorflow as tf
from modules import *
from data_load import *
from hyperparams_rl import Hyperparams as hp
import numpy as np
from tensorflow.contrib import slim
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os
import argparse
import math
def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    y = np.array(x == 3, dtype=np.int)
    final = []
    for slic in y:
        index = np.argmax(slic)
        if index == 0:
            temp = np.ones_like(slic, dtype=np.float)
            final.append(temp)
        else:
            slic[:index] = 1.0
            slic[index + 1:] = 0.0
            final.append(slic)
    return np.array(final, dtype=np.float32)

class Graph():
    def __init__(self, is_training=True,beam_width=5,mode="de2en"):
        self.graph = tf.Graph()
        self.is_training = is_training
        self.beam_width = beam_width
        de2idx, idx2de = load_de_vocab()
        en2idx, idx2en = load_en_vocab()
        with self.graph.as_default():
            if mode=="de2en":
                self.encoder_2idx = de2idx
                self.decoder_2idx = en2idx
            else:
                self.encoder_2idx = en2idx
                self.decoder_2idx = de2idx
            self.lr = tf.placeholder(tf.float32)
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))  # 注意这个是标准集里的
            self.is_inference = tf.placeholder(tf.bool)
            self.dropout_rate_tran = tf.placeholder(tf.float32)
            if is_training:
                self.lstm_drop_rate = tf.placeholder(tf.float32)
                self.dropout_rate = tf.placeholder(tf.float32)
                self.weight_initializer = tf.contrib.layers.xavier_initializer()
                self._selector = True
                self.image = tf.placeholder(tf.float32, shape=[None, 196, 1024])
                self.batch_size = tf.shape(self.image)[0]
                # sample
                self.preds_sample = self.image_caption(None, None, set="{}_caption".format(mode[:2]), flag=0)
                self.index = tf.placeholder(tf.bool, shape=[None])
                self.sample_index = tf.cast(self.index, tf.float32)
                self.not_sample_index = tf.cast(~self.index, tf.float32)
                # prob
                pad = tf.zeros([self.batch_size, 1], dtype=tf.int32)
                self.x_target = tf.concat([self.x[:, 1:], pad], axis=1)
                self.x_istarget = tf.to_float(tf.not_equal(self.x_target, 0))
                self.prob_z2x = -self.image_caption(self.x_istarget, self.x_target, set="{}_caption".format(mode[:2]),
                                                   flag=1)
            else:
                self.batch_size = tf.shape(self.x)[0]
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(self.encoder_2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=self.dropout_rate_tran,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=self.dropout_rate_tran,
                                                       is_training=is_training,
                                                       causality=False)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            #train translator
            if is_training:
                self.enc2dec = self.enc
                self.y_target =  tf.concat((self.y[:,1:],tf.zeros((self.batch_size,1),dtype=tf.int32)),axis=1)
                self.y_istarget = tf.to_float(tf.not_equal(self.y_target, 0))
                # reward
                _, self.logits_reward, _ = self.model(self.y)
                temp = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_reward,
                                                                       labels=tf.one_hot(self.y_target, len(self.decoder_2idx)))
                self.reward = -tf.reduce_sum(temp*self.y_istarget,axis=1)
                self.reward = self.reward - tf.reduce_min(self.reward)#baseline
                tf.stop_gradient(self.reward)
                #
                _,self.logits,self.y_preds = self.model(self.y)
                prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                       labels=tf.one_hot(self.y_target, len(self.decoder_2idx)))
                self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.y_preds, self.y_target))*self.y_istarget)/ (tf.reduce_sum(self.y_istarget))
                self.prob_x2y = tf.reduce_sum(prob*self.y_istarget,axis=1)
                # restore

                self.value_list = []
                self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model"))
                self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder"))

                self.value_list_de, self.value_list_en = [], []
                self.value_list_de.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="de_caption"))
                self.value_list_en.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="en_caption"))
                # optimize
                lamda = 1.0
                self.loss_rl = tf.reduce_sum(self.reward*self.prob_z2x*self.sample_index)/tf.reduce_sum(self.x_istarget*
                                                                                        tf.expand_dims(self.sample_index,axis=1))
                self.loss_nmt = tf.reduce_sum(self.prob_x2y*self.sample_index)/tf.reduce_sum(self.y_istarget*
                                                                                        tf.expand_dims(self.sample_index,axis=1))
                self.loss_fix = tf.reduce_sum(lamda*self.prob_z2x*self.not_sample_index)/tf.reduce_sum(self.x_istarget*
                                                                                        tf.expand_dims(self.not_sample_index,axis=1))
                self.loss = self.loss_nmt+self.loss_rl+self.loss_fix
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)#var_list=self.value_list
            else:
                # beam inference
                self.preds = tf.ones((self.batch_size, 1), dtype=tf.int32) * 2
                self.preds = self.beam_search(self.preds, self.beam_width, len(self.decoder_2idx))
                _,self.logits,_ = self.model(self.y)
                self.y_target = tf.concat((self.y[:,1:],tf.zeros((self.batch_size,1),dtype=tf.int32)),axis=1)
                self.y_istarget = tf.to_float(tf.not_equal(self.y_target, 0))
                temp = tf.reduce_sum(tf.log(tf.nn.softmax(self.logits))*tf.one_hot(self.y_target, len(self.decoder_2idx)),axis=2)#prob
                #self.prob = tf.reduce_sum(temp * self.y_istarget, axis=1)
                self.prob = tf.reduce_sum(temp*self.y_istarget,axis=1)/tf.reduce_sum(self.y_istarget,axis=1)
                self.value_list = []
                self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model"))
                self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder"))

    def beam_search(self, sentence, beam_width, num_classes):
        def true_fn():
            temp = tf.tile(tf.expand_dims(self.enc, axis=1), [1, self.beam_width, 1, 1])
            return tf.reshape(temp,[self.batch_size*self.beam_width,hp.maxlen,hp.hidden_units])
        self.enc2dec = tf.cond(self.is_inference,true_fn,
                           lambda: self.enc)
        sentence = tf.tile(tf.expand_dims(sentence,axis=1),[1,beam_width,1])
        #prob before
        value =tf.log([[1.] + [0.] * (beam_width - 1)])
        mask = tf.ones((self.batch_size,beam_width))
        for i in range(hp.maxlen-1):
            input = tf.reshape(sentence,[self.batch_size*beam_width,i+1])
            _,logits,_ = self.model(input)
            logits = logits[:,i,:]
            logits = tf.nn.log_softmax(tf.reshape(logits,[self.batch_size,beam_width,num_classes]))
            sum_logprob = tf.expand_dims(value, axis=2) + logits*tf.expand_dims(mask,axis=2)
            value, index = tf.nn.top_k(tf.reshape(sum_logprob, [self.batch_size, beam_width * num_classes]), k=beam_width)#batch x beam
            ids = index%num_classes#batch x beam
            pre_ids = index//num_classes#batch x beam
            pre_sentence = tf.batch_gather(sentence, pre_ids)#batch x beam x len
            new_word = tf.expand_dims(ids,axis=2)#batch x beam x 1
            sentence = tf.concat([pre_sentence,new_word],axis=2)#batch x beam x (len+1)
            mask = tf.batch_gather(mask,pre_ids)*tf.to_float(tf.not_equal(ids,3)) #第一项表示之前结束没，第二项表示现在结束了吗(0表示结束)
        preds = self.select(sentence,value)
        #复原
        self.enc2dec = self.enc
        return preds

    def select(self,sentence,value,alpha=1.0):
        beam_width = sentence.shape.as_list()[1]
        input = tf.reshape(sentence,[self.batch_size*beam_width,-1])
        mask = tf.py_func(my_func,[input],tf.float32)
        length = tf.reshape(tf.reduce_sum(mask,axis=1),[self.batch_size,beam_width])
        index = tf.expand_dims(tf.argmax(value/tf.pow(length,alpha),axis=1),axis=1)
        index = tf.cast(index,dtype=tf.int32)
        preds = tf.batch_gather(sentence,index)
        return tf.squeeze(preds)

    def model(self, decoder_inputs):

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(decoder_inputs,
                                     vocab_size=len(self.decoder_2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed",
                                     reuse=tf.AUTO_REUSE)

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(decoder_inputs,
                                                    #   vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe",
                                                    reuse=tf.AUTO_REUSE)
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                                                  [tf.shape(decoder_inputs)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe",
                                          reuse=tf.AUTO_REUSE)

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=self.dropout_rate_tran,
                                             training=tf.convert_to_tensor(self.is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=self.dropout_rate_tran,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention",
                                                       reuse=tf.AUTO_REUSE)

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc2dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=self.dropout_rate_tran,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       scope="vanilla_attention",
                                                       reuse=tf.AUTO_REUSE)

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units],
                                               reuse=tf.AUTO_REUSE)
                dec1 = tf.layers.dense(self.dec, len(self.decoder_2idx))
            dec2 = tf.to_int32(tf.arg_max(dec1, dimension=-1))
            return self.dec, dec1, dec2

    def beam_search_image(self,sentence, beam_width, num_classes):
        self.feature_beam = tf.tile(tf.expand_dims(self.feature, axis=1), [1, beam_width, 1, 1])
        sentence = tf.tile(tf.expand_dims(sentence, axis=1), [1, beam_width, 1])  # ba x beam x hidden
        total_sentence = tf.ones((self.batch_size, beam_width, 1), dtype=tf.int32) * 2
        self.last_state = [tf.reshape(tf.tile(tf.expand_dims(state, axis=1), [1, beam_width, 1]), [-1, hp.lstm_units])
                           for state in self.last_state]  # (ba x beam)x lstm_dim
        self.last_output = tf.tile(tf.expand_dims(self.last_output, axis=1), [1, beam_width, 1])  # 同上
        value = tf.log([[1.] + [0.] * (beam_width - 1)])
        mask = tf.ones((self.batch_size, beam_width))
        for i in range(hp.maxlen - 1):
            alpha = self.attention(self.last_output,self.feature_beam)  # ba x beam x 196
            image_attention = tf.reduce_sum(self.feature_beam * tf.expand_dims(alpha, -1),
                                            axis=-2)  # batch_size x beam x 1024
            if self._selector:
                image_attention = self.selector(image_attention, self.last_output)

            inputs = tf.reshape(tf.concat((image_attention, sentence), axis=-1), [-1, hp.hidden_units_cap + 1024])
            output, state = self.lstm(inputs, self.last_state)
            output = tf.reshape(output, [self.batch_size, beam_width, hp.lstm_units])

            expanded_output = tf.concat([output,
                                         sentence,
                                         image_attention],
                                        axis=-1)  # ba x beam x sth
            logits = self.decode(expanded_output)
            logits = tf.nn.log_softmax(logits)
            sum_logprob = tf.expand_dims(value, axis=2) + logits * tf.expand_dims(mask, axis=2)
            t = tf.reshape(sum_logprob, [-1, beam_width * num_classes])
            value, index = tf.nn.top_k(t, k=beam_width)  # batch x beam
            ids = index % num_classes  # batch x beam
            pre_ids = index // num_classes  # batch x beam

            sentence = tf.nn.embedding_lookup(self.lookup_table, ids)
            pre_sentence = tf.batch_gather(total_sentence, pre_ids)  # batch x beam x len

            new_word = tf.expand_dims(ids, axis=2)  # batch x beam x 1
            total_sentence = tf.concat([pre_sentence, new_word], axis=2)  # batch x beam x (len+1)
            mask = tf.batch_gather(mask, pre_ids) * tf.to_float(tf.not_equal(ids, 3))  # 第一项表示之前结束没，第二项表示现在结束了吗(0表示结束)
            # 下一循环要用的
            self.last_output = output
            self.last_state = state
        preds = self.select(total_sentence, value)
        return preds

    def image_caption(self, mask, preds, set,flag=1):
        with tf.variable_scope(set,reuse=tf.AUTO_REUSE):
            with tf.variable_scope("embedding"):
                lookup_table = tf.get_variable('lookup_table',
                                               dtype=tf.float32,
                                               shape=[len(self.encoder_2idx), hp.hidden_units_cap],
                                               initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                self.lookup_table = lookup_table
            with tf.variable_scope("lstm"):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(hp.lstm_units)
                lstm = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,
                    input_keep_prob=1.0 - self.lstm_drop_rate,
                    output_keep_prob=1.0 - self.lstm_drop_rate)
                self.lstm = lstm
            self.feature = tf.contrib.layers.batch_norm(inputs=self.image,
                                                        decay=0.95,
                                                        center=True,
                                                        scale=True, updates_collections=None,
                                                        is_training=~self.is_inference)
            with tf.variable_scope("initialize"):
                context_mean = tf.reduce_mean(self.feature, axis=1)
                initial_memory, initial_output = self.initial(context_mean)
                initial_state = initial_memory, initial_output
            last_state = initial_state
            last_output = initial_output
            self.last_state, self.last_output = initial_state, initial_output
            alpha_list,prob_list = [],[]
            # sample
            sentence = tf.nn.embedding_lookup(lookup_table, tf.ones((self.batch_size), dtype=tf.int32) * 2)
            if flag==0:
                self.sample = self.beam_search_image(sentence, beam_width=self.beam_width, num_classes=len(self.encoder_2idx))
                return self.sample
            else:
                inputs_sentences = tf.nn.embedding_lookup(lookup_table, preds)
                #get the probility
                for i in range(hp.maxlen):
                    #input
                    alpha = self.attention(last_output,self.feature)  # batch_size x 196
                    alpha_list.append(alpha)
                    image_attention = tf.reduce_sum(self.feature * tf.expand_dims(alpha, 2), axis=1)  # batch_size x 1024
                    if self._selector:
                        image_attention = self.selector(image_attention, last_output)
                    inputs = tf.concat((image_attention, sentence), axis=1)
                    #output
                    output, state = lstm(inputs, last_state)
                    temp = tf.layers.dropout(output, rate=self.dropout_rate)
                    #decode
                    expanded_output = tf.concat([temp,
                                                 sentence,
                                                 image_attention],
                                                axis=1)
                    logits = self.decode(expanded_output)
                    p = tf.log(tf.nn.softmax(logits))
                    t = tf.reduce_sum(p*tf.one_hot(preds[:,i],len(self.encoder_2idx)),axis=1)
                    prob_list.append(t)
                    #next step
                    sentence = inputs_sentences[:, i, :]  # batch_size x embed_dim
                    last_state = state
                    last_output = output
                prob = tf.reduce_sum(tf.stack(prob_list,axis=1)*mask,axis=1)
                return prob

    def decode(self,expanded_output):
        with tf.variable_scope("decode"):
            temp1 = tf.layers.dense(expanded_output,hp.decode_layerunit,name="decode_fc_1",
            activation = tf.tanh,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)
            temp2 = tf.layers.dropout(temp1,rate=self.dropout_rate)
            logits = tf.layers.dense(temp2,
                                    units = len(self.encoder_2idx),
                                    activation = None,
                                    name = 'decode_fc_2',reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)
            return logits

    def selector(self, context, h):
        beta = tf.layers.dense(h, 1, kernel_initializer=self.weight_initializer, name="selector", activation=tf.sigmoid,
                               reuse=tf.AUTO_REUSE)
        context = tf.multiply(beta, context, name='selected_context')
        return context

    def initial(self, context_mean):
        # 可以加dropout，暂时没加
        output = tf.layers.dense(context_mean, hp.lstm_units, activation=tf.tanh,
                                 kernel_initializer=self.weight_initializer)
        # initialize

        # output = tf.layers.dense(temp1, hp.lstm_units)

        memory = tf.layers.dense(context_mean, hp.lstm_units, activation=tf.tanh,
                                 kernel_initializer=self.weight_initializer)

        # memory = tf.layers.dense(temp2, hp.lstm_units)

        return memory, output

    def attention(self,output,feature):
        # reshaped_images = tf.reshape(self.feature, [-1, 1024])
        #output可以dropout
        temp1 = tf.layers.dense(feature,
                                  units = hp.attention_dim,name="fc_1",
                                  activation = None,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer
                                  )#ba x beam x 196 x att
        temp2 = tf.layers.dense(output,
                              units = hp.attention_dim,name="fc_2",
                              activation = None,reuse=tf.AUTO_REUSE,use_bias=None,kernel_initializer =self.weight_initializer
                              )#ba x beam x att
        temp2 = tf.expand_dims(temp2, axis=-2)
        # temp2 = tf.reshape(temp2, [-1, hp.attention_dim])
        temp = tf.nn.relu(temp1 + temp2)
        logits = tf.layers.dense(temp,
                               units = 1,
                               activation = None,name="fc_3",
                               use_bias = False,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)
        #ba x beam x 196 x 1
        logits = tf.squeeze(logits,axis=-1)
        # logits = tf.reshape(logits, [-1, 196])
        alpha = tf.nn.softmax(logits)
        return alpha

def process(preds):
    y = np.array(np.array(preds) == 3, dtype=np.int)
    final = []
    for slic in y:
        index = np.argmax(slic)
        if index == 0:
            temp = np.ones_like(slic, dtype=np.int)
            final.append(temp)
        else:
            slic[:index] = 1
            slic[index+1:] = 0
            final.append(slic)
    return np.array(final, dtype=np.int32)

def mix(preds,y,x_ratio=0.5):
    num = len(y)
    preds, y = np.array(preds), np.array(y)
    sample_index = np.ones(shape=(num), dtype=np.int32)
    r_index = np.random.permutation(num)[:round(num * x_ratio)]
    preds[r_index] = y[r_index]
    sample_index[r_index] = 0
    return preds, sample_index

def concat(name):
    new_name = name+"1"
    os.system(r"sed -r 's/(@@ )|(@@ ?$)//g' < {} > {} ".format(name, new_name))
    os.system("rm {}".format(name))
    os.system("mv {} {}".format(new_name, name))

image_path = "../../image/flickr30k_ResNets50_blck4_{}.fp16.npy"
def train():
    # argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default="de2en")
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--save_dir', type=str, default="result")
    parser.add_argument('--save_file', type=str, default="bleu.txt")
    parser.add_argument('--save_log', type=str, default="logdir")
    parser.add_argument('--task', type=str, default="task2")
    parser.add_argument('--set', type=str, default="val")
    args = parser.parse_args()
    mode = args.mode
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_path = args.save_log
    save_file = args.save_file
    save_dir = args.save_dir
    task = args.task
    set = args.set
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not os.path.exists(save_dir):os.mkdir(save_dir)
    # prepare
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    translator = eval("idx2{}".format(mode[-2:]))  # !!!
    # load_graph
    g = Graph(is_training=True, beam_width=1, mode=mode)
    g_val = Graph(is_training=False, beam_width=5, mode=mode)
    print("Graph loaded")
    # Load data
    X, Image_index, Y, Targets = load_rl_data(language=mode[:2])
    images = np.load(image_path.format("train"))
    num_batch = int(math.ceil(len(X) / hp.batch_size))
    x_val, Targets_val,idents = load_test_rl_data(set=set, task=task, language=mode[:2])
    num_batch_val = int(math.ceil(len(x_val) / hp.batch_size_test))

    # prepare ref file
    if task == "task2":
        f_ref1 = open("{}/{}_ref_1".format(save_dir,mode), "w+")
        f_ref2 = open("{}/{}_ref_2".format(save_dir,mode), "w+")
        f_ref3 = open("{}/{}_ref_3".format(save_dir,mode), "w+")
        f_ref4 = open("{}/{}_ref_4".format(save_dir,mode), "w+")
        f_ref5 = open("{}/{}_ref_5".format(save_dir,mode), "w+")
        f_ref = [f_ref1, f_ref2, f_ref3, f_ref4, f_ref5]
    else:
        f_ref1 = open("{}/{}_ref_1".format(save_dir,mode), "w+")
        f_ref = [f_ref1]
    for i in range(len(Targets_val)):
        for k, l in enumerate(f_ref):
            if task == "task2":
                l.write(Targets_val[i][k] + "\n")
            else:
                l.write(Targets_val[i] + "\n")
    for sth in f_ref:
        sth.close()
        concat(sth.name)
    val_sum = len(Targets_val)
    temp = []
    for file in f_ref:
        with open(file.name,"r") as h:
            refs = [i.strip() for i in h.readlines()]
            temp.extend(refs)
    Target_val_split = []
    for i in range(val_sum):
        temp1 = []
        for j in range(len(f_ref)):
            temp1.append(temp[i+j*val_sum].split())
        Target_val_split.append(temp1)
    save_num = 1
    best_of_now = 0.0
    pre_bleu = 0.0
    # saver
    saver1 = tf.train.Saver(var_list=g.value_list,max_to_keep=100)
    if mode == "de2en":
        saver2 = tf.train.Saver(var_list=g.value_list_de)
    else:
        saver2 = tf.train.Saver(var_list=g.value_list_en)
    saver_val = tf.train.Saver(var_list=g_val.value_list)
    # config
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=g.graph) as sess:
        ## Restore parameters
        sess.run(tf.global_variables_initializer())
        if mode == "de2en":
            # saver1.restore(sess, "../rl/{}".format(mode) + "/step_1430")
            saver1.restore(sess, "{}".format(mode) + "/step_8000")
            saver2.restore(sess, eval("hp.logdir_cap_{}".format(mode[:2])) + "/model_step_9999")  # "/model_step_3999"
        elif mode == "en2de":
            # saver1.restore(sess, "../rl/{}".format(mode)+ "/step_3363")
            saver1.restore(sess, "{}".format(mode) + "/step_7000")
            saver2.restore(sess, eval("hp.logdir_cap_{}".format(mode[:2])) + "/model_step_3999")
        print("Restored!")
        lr = hp.lr
        num = 0
        for epoch in range(hp.num_epochs):
            for i in range(num_batch):
                step = epoch * num_batch + i+1
                lr = hp.lr#* min(pow(step, -0.5),step * pow(hp.warmup_step, -1.5))
                image = images[Image_index[i * hp.batch_size: (i + 1) * hp.batch_size]]
                x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]
                if step% 100 == 0:#!!
                    # prepare save file
                    f_hypo = open("{}/{}_hypo_{}".format(save_dir,mode, save_num), "w+")
                    save_num += 1
                    # save log
                    saver1.save(sess, save_path + "/step_{}".format(step))
                    # write file
                    with tf.Session(graph=g_val.graph) as sess_val:
                        sess_val.run(tf.global_variables_initializer())
                        saver_val.restore(sess_val, tf.train.latest_checkpoint(save_path))
                        for j in range(num_batch_val):
                            # cal_pred
                            x_val_batch = x_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]
                            feed_dict = {
                                g_val.x: x_val_batch,
                                g_val.dropout_rate_tran: 0.0,
                                g_val.is_inference: True
                            }
                            preds = sess_val.run(g_val.preds, feed_dict)
                            preds = np.concatenate((preds[:, 1:], np.zeros((len(x_val_batch), 1))), axis=1)
                            # corporate
                            for pred in preds:  # sentence-wisex_val_batch
                                got = " ".join(translator[idx] for idx in pred).split("</S>")[0].strip()
                                f_hypo.write(got + "\n")
                        f_hypo.close()
                        concat(f_hypo.name)
                        with open(f_hypo.name,"r") as f:
                            hypos = [i.strip().split() for i in f.readlines()]
                            n_bleu = corpus_bleu(Target_val_split, hypos, smoothing_function=SmoothingFunction().method2)
                            with open(save_file,"a+") as h:
                                h.write(str(n_bleu)+"\n")
                        #     if n_bleu>best_of_now:
                        #         best_of_now = n_bleu
                        #         num = 0
                        #     elif n_bleu < pre_bleu:
                        #         num += 1
                        #         if num==2:
                        #             lr = lr*0.7
                        #             num = 0
                        #     pre_bleu = n_bleu
                #sample from image
                feed_dict_sample = {
                    g.dropout_rate:0.0,
                    g.lstm_drop_rate: 0.0,
                    g.image: image,
                    g.is_inference: True
                }
                preds = sess.run(g.preds_sample, feed_dict_sample)
                preds = process(preds) * preds
                mix_x, sample_index = mix(preds, x, x_ratio=0.5)  # !!!!!
                #train
                feed_dict = {
                    g.x: mix_x,
                    g.image: image,
                    g.dropout_rate: hp.dropout_rate,
                    g.dropout_rate_tran: hp.dropout_rate_tran,
                    g.lstm_drop_rate: hp.lstm_drop_rate,
                    g.index: sample_index,
                    g.y: y,
                    g.is_inference: False,
                    g.lr:lr
                }
                _, loss = sess.run([g.train_op, g.loss], feed_dict)
                print(loss)
                # if (step + 1) % 100 == 0:
                #     with  open(save_file, "a+") as f:
                #         f.write("loss\n")
                #         f.write(str(loss) + "\n")

if __name__ == '__main__':
    train()
    print("Done")


