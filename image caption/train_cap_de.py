from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import load_cap_data, load_de_vocab, load_en_vocab
from nltk.translate.bleu_score import SmoothingFunction
from tensorflow.contrib import slim
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
    def __init__(self,is_training):
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._selector = True
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.x_target = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            self.image = tf.placeholder(tf.float32,shape=[None,196,1024])
            self.dropout_rate = tf.placeholder(tf.float32)
            self.lstm_drop_rate = tf.placeholder(tf.float32)
            self.lr = tf.placeholder(tf.float32,shape=[])
            batch_size = tf.shape(self.image)[0]
            self.batch_size = batch_size
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
            self.en2idx = en2idx
            self.de2idx = de2idx
            self.weight_initializer = tf.contrib.layers.xavier_initializer()
            self.istarget = tf.to_float(tf.not_equal(self.x_target, 0))
            with tf.variable_scope("de_caption"):
                with tf.variable_scope("embedding"):
                    lookup_table = tf.get_variable('lookup_table',
                                        dtype=tf.float32,
                                        shape=[len(self.de2idx), hp.hidden_units_cap],
                                        initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
                    self.lookup_table = lookup_table
                with tf.variable_scope("lstm"):
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(hp.lstm_units)

                    #lstm = lstm_cell
                    lstm = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,
                    input_keep_prob = 1.0-self.lstm_drop_rate,
                    output_keep_prob = 1.0-self.lstm_drop_rate)
                    self.lstm = lstm
                self.feature = tf.contrib.layers.batch_norm(inputs=self.image,
                                            decay=0.95,
                                            center=True,
                                            scale=True,updates_collections=None,is_training=self.is_training)
                with tf.variable_scope("initialize"):
                    context_mean = tf.reduce_mean(self.feature, axis = 1)
                    initial_memory, initial_output = self.initial(context_mean)
                    initial_state = initial_memory, initial_output
                last_state = initial_state
                last_output = initial_output
                self.last_state, self.last_output = initial_state, initial_output
                logit_list,self.preds_list,alpha_list = [],[],[]
                sentence = tf.nn.embedding_lookup(lookup_table,tf.ones(batch_size,dtype=tf.int32)*2)
                if not is_training:
                    beam_width = 5
                    self.feature = tf.tile(tf.expand_dims(self.feature, axis=1), [1, beam_width, 1, 1])
                    self.preds = self.beam_search(sentence, beam_width=beam_width, num_classes=len(de2idx))
                else:
                    for i in range(hp.maxlen):
                        #batch_size x embed_dim
                        alpha = self.attention(last_output)#batch_size x 196
                        mask_alpha = tf.tile(tf.expand_dims(self.istarget[:, i], 1),
                                             [1, 196])
                        alpha_list.append(alpha*mask_alpha)

                        image_attention = tf.reduce_sum(self.feature*tf.expand_dims(alpha,-1),axis=-2)#batch_size x 1024
                        if self._selector:
                            image_attention = self.selector(image_attention,last_output)
                        inputs = tf.concat((image_attention,sentence),axis=-1)
                        output,state = lstm(inputs,last_state)
                        #!!
                        temp = tf.layers.dropout(output,rate=self.dropout_rate)
                        expanded_output = tf.concat([temp,
                                                    sentence,
                                                    image_attention],
                                                    axis = -1)
                        logits = self.decode(expanded_output)
                        prediction = tf.argmax(logits, 1)
                        self.preds_list.append(prediction)
                        logit_list.append(logits)
                        sentence = tf.nn.embedding_lookup(lookup_table,self.x[:,i])
                        last_state = state
                        last_output = output
            if is_training:
                self.preds_list = tf.stack(self.preds_list,axis=1)
                logits = tf.stack(logit_list,axis=1)
                alpha_list = tf.stack(alpha_list,axis=1)
                attentions = tf.reduce_sum(alpha_list,axis=1)
                diffs = tf.ones_like(attentions) - attentions
                attention_loss = hp.attention_loss_factor \
                                 * tf.nn.l2_loss(diffs) \
                                 / tf.cast((batch_size * 196),dtype=tf.float32)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.x_target,len(de2idx)),logits=logits)
                self.loss = tf.reduce_sum(self.loss*self.istarget)/tf.reduce_sum(self.istarget)+attention_loss
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
                self.global_step = tf.Variable(0,
                                           name = 'global_step',
                                           trainable = False)
                # self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss = self.loss,
                    global_step = self.global_step,
                    learning_rate = self.lr,
                    optimizer = self.optimizer,
                    clip_gradients = hp.clip_gradients)
            self.value_list = slim.get_variables_to_restore()

    def beam_search(self, sentence, beam_width, num_classes):
        sentence = tf.tile(tf.expand_dims(sentence,axis=1),[1,beam_width,1])#ba x beam x hidden
        total_sentence = tf.ones((self.batch_size,beam_width,1),dtype=tf.int32) * 2
        self.last_state = [tf.reshape(tf.tile(tf.expand_dims(state,axis=1),[1,beam_width,1]),[-1,hp.lstm_units]) for state in self.last_state]#(ba x beam)x lstm_dim
        self.last_output = tf.tile(tf.expand_dims(self.last_output,axis=1),[1,beam_width,1])#同上
        value = tf.log([[1.] + [0.] * (beam_width - 1)])
        mask = tf.ones((self.batch_size,beam_width))
        for i in range(hp.maxlen-1):
            alpha = self.attention(self.last_output)#ba x beam x 196
            image_attention = tf.reduce_sum(self.feature * tf.expand_dims(alpha, -1), axis=-2)  # batch_size x beam x 1024
            if self._selector:
                image_attention = self.selector(image_attention, self.last_output)

            inputs = tf.reshape(tf.concat((image_attention,sentence),axis=-1),[-1,hp.hidden_units_cap+1024])
            output, state = self.lstm(inputs,self.last_state)
            output = tf.reshape(output,[self.batch_size,beam_width,hp.lstm_units])

            expanded_output = tf.concat([output,
                                         sentence,
                                         image_attention],
                                        axis=-1)#ba x beam x
            logits = self.decode(expanded_output)
            logits = tf.nn.log_softmax(logits)
            sum_logprob = tf.expand_dims(value, axis=2) + logits*tf.expand_dims(mask,axis=2)
            value, index = tf.nn.top_k(tf.reshape(sum_logprob, [self.batch_size, beam_width * num_classes]), k=beam_width)#batch x beam
            ids = index%num_classes#batch x beam
            pre_ids = index//num_classes#batch x beam

            sentence = tf.nn.embedding_lookup(self.lookup_table, ids)
            pre_sentence = tf.batch_gather(total_sentence, pre_ids)#batch x beam x len

            new_word = tf.expand_dims(ids,axis=2)#batch x beam x 1
            total_sentence = tf.concat([pre_sentence,new_word],axis=2)#batch x beam x (len+1)
            mask = tf.batch_gather(mask,pre_ids)*tf.to_float(tf.not_equal(ids,3)) #第一项表示之前结束没，第二项表示现在结束了吗(0表示结束)
            #下一循环要用的
            self.last_output = output
            self.last_state = state
        preds = self.select(total_sentence,value)
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

    def selector(self,context,h):
        beta = tf.layers.dense(h,1,kernel_initializer =self.weight_initializer,name="selector",activation=tf.sigmoid,reuse=tf.AUTO_REUSE)
        context = tf.multiply(beta, context, name='selected_context')
        return context

    def decode(self,expanded_output):
        with tf.variable_scope("decode"):
            temp1 = tf.layers.dense(expanded_output,hp.decode_layerunit,name="decode_fc_1",
            activation = tf.tanh,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)#!!!!!!!!!!!!
            temp2 = tf.layers.dropout(temp1,rate=self.dropout_rate)
            logits = tf.layers.dense(temp2,
                                    units = len(self.de2idx),
                                    activation = None,
                                    name = 'decode_fc_2',reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)
            return logits
    def initial(self,context_mean):
        #可以加dropout，暂时没加
        output = tf.layers.dense(context_mean, hp.lstm_units,activation=tf.tanh,kernel_initializer =self.weight_initializer)
        #initialize

        # output = tf.layers.dense(temp1, hp.lstm_units)

        memory = tf.layers.dense(context_mean, hp.lstm_units,activation=tf.tanh,kernel_initializer =self.weight_initializer)

        # memory = tf.layers.dense(temp2, hp.lstm_units)

        return memory,output
    def attention(self,output):
        # reshaped_images = tf.reshape(self.feature, [-1, 1024])
        #output可以dropout
        temp1 = tf.layers.dense(self.feature,
                                  units = hp.attention_dim,name="fc_1",
                                  activation = None,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer
                                  )
        temp2 = tf.layers.dense(output,
                              units = hp.attention_dim,name="fc_2",
                              activation = None,reuse=tf.AUTO_REUSE,use_bias=None,kernel_initializer =self.weight_initializer
                              )
        temp2 = tf.expand_dims(temp2, axis=-2)
        # temp2 = tf.reshape(temp2, [-1, hp.attention_dim])
        temp = tf.nn.relu(temp1 + temp2)
        logits = tf.layers.dense(temp,
                               units = 1,
                               activation = None,name="fc_3",
                               use_bias = False,reuse=tf.AUTO_REUSE,kernel_initializer =self.weight_initializer)

        logits = tf.squeeze(logits,axis=-1)
        # logits = tf.reshape(logits, [-1, 196])
        alpha = tf.nn.softmax(logits)
        return alpha

    
                
image_path = "../../image/flickr30k_ResNets50_blck4_{}.fp16.npy"
def train(): 
    # Load graph
    g = Graph(is_training=True)
    print("Graph loaded")
    # Load data
    X, Image_index,_,X_target= load_cap_data(set="de")
    images = np.load(image_path.format("train"))
    num_batch = int(math.ceil(len(X) / hp.batch_size))
    if not os.path.exists(hp.logdir_cap_de): os.mkdir(hp.logdir_cap_de)
    # Start session         
    with g.graph.as_default():   
        saver = tf.train.Saver(var_list=g.value_list,max_to_keep=40)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, tf.train.latest_checkpoint("logdir_en2"))
            # print("Restored!")
            ## train
            for epoch in range(hp.num_epochs):
                for i in range(num_batch):
                    lr = hp.lr_cap*pow(0.95,epoch)
                    step = epoch*num_batch+i
                    ### Get mini-batches
                    image = images[Image_index[i*hp.batch_size: (i+1)*hp.batch_size]]
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    x_target = X_target[i*hp.batch_size: (i+1)*hp.batch_size]
                    feed_dict= {
                        g.x: x, 
                        g.image:image,
                        g.dropout_rate:hp.dropout_rate,
                        g.lstm_drop_rate:hp.lstm_drop_rate,
                        g.lr:lr,
                        g.x_target:x_target
                    }
                    if i%1000==0:
                       _,loss,preds=sess.run([g.train_op,g.loss,g.preds_list],feed_dict)
                       with open("de.txt","a+") as f:
                         f.write("loss {}".format(i)+" "+str(loss))
                    else:
                       sess.run(g.train_op,feed_dict)
                    if (step+1)%1000==0:
                        saver.save(sess,save_path=hp.logdir_cap_de+ '/model_step_%d'%step)

                                          
if __name__ == '__main__':
    train()
    print("Done")
    




