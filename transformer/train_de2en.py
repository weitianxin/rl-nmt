from __future__ import print_function
import os
from hyperparams import Hyperparams as hp
from data_load import *
from modules import *
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import tensorflow as tf
import numpy as np
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
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        self.is_training = is_training
        with self.graph.as_default():
            if is_training:
                # self.x, self.y,self.num_batch = get_batch_data() # (N, T)
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y_target = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.lr = tf.placeholder(tf.float32)
                self.dropout_rate = tf.placeholder(tf.float32)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, 1))
                self.dropout_rate = tf.placeholder(tf.float32)
                # self.image = tf.placeholder(tf.float32,shape=[None,196,1024])
            # define decoder inputs
            self.batch_size = tf.shape(self.x)[0]
            # Load vocabulary    
            de2idx, _= load_de_vocab()
            en2idx, _ = load_en_vocab()
            self.de2idx = de2idx
            self.en2idx = en2idx
            # Encoder
            
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x, 
                                      vocab_size=len(de2idx), 
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
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=self.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            
            # Decoder
            if is_training:
                _,self.logits,self.preds = self.model(self.y)#batch x max_len
            else:#!!
                self.preds = self.beam_search(self.y,beam_width=1,num_classes=len(en2idx))
            self.value_list = []
            self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="model"))
            self.value_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="encoder"))
            if is_training:  
                #acc
                self.istarget = tf.to_float(tf.not_equal(self.y_target, 0))
                self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y_target))*self.istarget)/ (tf.reduce_sum(self.istarget))
                tf.summary.scalar('acc', self.acc)
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y_target, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.99, epsilon=1e-9)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

    def model(self, decoder_inputs):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(decoder_inputs,
                                     vocab_size=len(self.en2idx),
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
                                             rate=self.dropout_rate,
                                             training=tf.convert_to_tensor(self.is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention",
                                                       reuse=tf.AUTO_REUSE)

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=self.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       scope="vanilla_attention",
                                                       reuse=tf.AUTO_REUSE)

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units],
                                               reuse=tf.AUTO_REUSE)
                dec1 = tf.layers.dense(self.dec, len(self.en2idx))
            dec2 = tf.to_int32(tf.arg_max(dec1, dimension=-1))
            return self.dec, dec1, dec2

    def beam_search(self, sentence, beam_width, num_classes):
        def true_fn():
            temp = tf.tile(tf.expand_dims(self.enc, axis=1), [1, beam_width, 1, 1])
            return tf.reshape(temp,[self.batch_size*beam_width,hp.maxlen,hp.hidden_units])
        self.enc = true_fn()
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
        return preds

    def select(self,sentence,value,alpha=0.7):
        beam_width = sentence.shape.as_list()[1]
        input = tf.reshape(sentence,[self.batch_size*beam_width,-1])
        mask = tf.py_func(my_func,[input],tf.float32)
        length = tf.reshape(tf.reduce_sum(mask,axis=1),[self.batch_size,beam_width])
        index = tf.expand_dims(tf.argmax(value/tf.pow(length,alpha),axis=1),axis=1)
        index = tf.cast(index,dtype=tf.int32)
        preds = tf.batch_gather(sentence,index)
        return tf.squeeze(preds)

if __name__ == '__main__':                
    # Load vocabulary    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Construct graph
    g = Graph(True)
    g_val = Graph(False);print("Graph loaded")
    # x_val,Targets = load_val_data()
    # num_batch_val = len(x_val)//hp.batch_size
    mode="de2en"
    task="task1"
    x,y,y_target = load_train_data(mode="de2en")
    num_batch = int(math.ceil(len(x)/hp.batch_size))
    smoothie = SmoothingFunction().method2
    x_val, Targets = load_val_data(mode=mode, task=task)

    num_batch_val = int(math.ceil(len(x_val) / hp.batch_size_test))#!!batch_test
    # Start session
    if not os.path.exists(hp.logdir_de2en):os.mkdir(hp.logdir_de2en)
    with g.graph.as_default():
        saver = tf.train.Saver(var_list=g.value_list,max_to_keep=100)
        saver_val = tf.train.Saver(var_list=g_val.value_list)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("start")
            #saver.restore(sess,tf.train.latest_checkpoint(hp.logdir))
            #print("restore")
            for epoch in range(hp.num_epochs):
                for i in range(num_batch):
                    step = i+1+epoch*num_batch
                    lr = hp.lr*min(pow(step,-0.5),step*pow(hp.warmup_step,-1.5))
                    #train
                    feed_dict={
                        g.x:x[i*hp.batch_size:(i+1)*hp.batch_size],
                        g.y:y[i*hp.batch_size:(i+1)*hp.batch_size],
                        g.y_target:y_target[i*hp.batch_size:(i+1)*hp.batch_size],
                        g.lr:lr,
                        g.dropout_rate:hp.dropout_rate
                    }
                    if i%10==0:
                       _,train_loss=sess.run([g.train_op,g.mean_loss],feed_dict)
                       print("epoch:"+str(epoch)+" step:"+str(i)+" train_loss:"+str(train_loss))
                    else:
                       sess.run(g.train_op,feed_dict)
                    #val
                    if (step+1)%1000==0:
                        saver.save(sess,save_path=hp.logdir_de2en + '/model_epoch_%d'%step)
                        with tf.Session(graph=g_val.graph) as sess_val:
                            sess_val.run(tf.global_variables_initializer())
                            saver_val.restore(sess_val, tf.train.latest_checkpoint(hp.logdir_de2en))
                            list_of_refs,hypotheses=[],[]
                            for j in range(num_batch_val):
                                targets = Targets[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]
                                preds = np.ones((len(targets), 1), dtype=np.int32) * 2
                                feed_dict_val = {
                                    g_val.x: x_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test],
                                    g_val.y: preds,
                                    g_val.dropout_rate: 0.0
                                }
                                preds = sess_val.run(g_val.preds, feed_dict_val)
                                preds = np.concatenate((preds[:, 1:], np.zeros((len(targets), 1))), axis=1)
                                # corporate
                                for target, pred in zip(targets, preds):  # sentence-wise
                                    got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                                    # bleu score
                                    # ref = target.split()
                                    ref = target.split()
                                    hypothesis = got.split()
                                    # if len(ref) > 3 and len(hypothesis) > 3:
                                    if len(ref)>3 and len(hypothesis) > 3:
                                        list_of_refs.append([ref])
                                        hypotheses.append(hypothesis)
                        with open(hp.output_file, "a+") as f:
                            n_bleu = corpus_bleu(list_of_refs, hypotheses, smoothing_function=smoothie)
                            f.write(" val bleu score:" + str(n_bleu) + "\n")
    print("Done")

# if (i + 1) % 1000 == 0:
#     list_of_refs, hypotheses = [], []
#     for j in range(num_batch_val):
#
#         # bleu score
#         # cal_pred
#         targets = Targets[j * hp.batch_size:(j + 1) * hp.batch_size]
#         preds = np.concatenate((np.ones((hp.batch_size, 1)) * 2, np.zeros((hp.batch_size, hp.maxlen - 1))), axis=1)
#         for k in range(hp.maxlen - 1):
#             # print(x_val[j*hp.batch_size:(j+1)*hp.batch_size].shape,preds.shape,y_target_val[j*hp.batch_size:(j+1)*hp.batch_size].shape)
#             feed_dict_val = {
#                 g.x: x_val[j * hp.batch_size:(j + 1) * hp.batch_size],
#                 g.y: preds,
#                 # g.y_target:y_target_val[j*hp.batch_size:(j+1)*hp.batch_size],
#                 g.lr: 0.0,
#                 g.dropout_rate: 0.0
#             }
#             _preds = sess.run(g.preds, feed_dict_val)
#             preds[:, k + 1] = _preds[:, k]
#
#     preds = np.concatenate((preds[:, 1:], np.zeros((hp.batch_size, 1))), axis=1)
#     # corporate
#     for target, pred in zip(targets, preds):  # sentence-wise
#         got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
#         # bleu score
#         # ref = target.split()
#         ref = [i.split() for i in target]
#         hypothesis = got.split()
#         # if len(ref) > 3 and len(hypothesis) > 3:
#         if all([len(sen) > 3 for sen in ref]) and len(hypothesis) > 3:
#             list_of_refs.append(ref)
#             hypotheses.append(hypothesis)
# with open(hp.output_file, "a+") as f:
#     n_bleu = corpus_bleu(list_of_refs, hypotheses, smoothing_function=smoothie)
#     f.write("train_loss:" + str(train_loss) + " val bleu score:" + str(n_bleu) + "\n")
