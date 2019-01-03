import os
from data_load import *
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import tensorflow as tf
from hyperparams import Hyperparams as hp
import numpy as np
import math,os
from train_de2en import Graph#!!!
mode = "de2en"#!!!
task="task1"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
if __name__ == '__main__':
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    translator = idx2en#!!!!!!!!
    save_list = os.listdir(eval("hp.logdir_{}".format(mode)))
    file_list = set(["/"+save.split('.')[0] for save in save_list if save!="checkpoint"])
    file_list = sorted(file_list,key=lambda x:int(x.split('_')[2]))
    g = Graph(False)
    print("Graph loaded")
    # Load vocabulary

    x_val, Targets = load_val_data(mode=mode,task=task)
    num_batch_val = int(math.ceil(len(x_val) / hp.batch_size_test))
    saver = tf.train.Saver(var_list=g.value_list)
    if task=="task2":
        f_ref1 = open("../result/transformer/task2/{}/ref_1".format(mode), "a+")
        f_ref2 = open("../result/transformer/task2/{}/ref_2".format(mode), "a+")
        f_ref3 = open("../result/transformer/task2/{}/ref_3".format(mode), "a+")
        f_ref4 = open("../result/transformer/task2/{}/ref_4".format(mode), "a+")
        f_ref5 = open("../result/transformer/task2/{}/ref_5".format(mode), "a+")
        f_ref = [f_ref1,f_ref2,f_ref3,f_ref4,f_ref5]
    else:
        f_ref1 = open("../result/transformer/task1/{}/ref1_1".format(mode), "a+")
        f_ref = [f_ref1]
    with g.graph.as_default():
        for i,file in enumerate(file_list):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,eval("hp.logdir_{}".format(mode))+file)#file
                if task == "task2":
                    f_hypo = open("../result/transformer/task2/{}/hypo_{}".format(mode, i + 1), "a+")
                    #!!!!!!
                else:
                    f_hypo = open("../result/transformer/task1/{}/hypo1_{}".format(mode, i + 1), "a+")
                print("restore")
                for j in range(num_batch_val):
                    # bleu score
                    # cal_pred
                    targets = Targets[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]
                    preds = np.ones((len(targets), 1),dtype=np.int32) * 2
                    feed_dict_val = {
                        g.x: x_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test],
                        g.y: preds,
                        g.dropout_rate: 0.0
                    }
                    preds = sess.run(g.preds, feed_dict_val)
                    preds = np.concatenate((preds[:, 1:], np.zeros((len(targets), 1))), axis=1)
                    # corporate
                    for target, pred in zip(targets, preds):  # sentence-wise
                        got = " ".join(translator[idx] for idx in pred).split("</S>")[0].strip()
                        if i==0:
                            for k,l in enumerate(f_ref):
                                l.write(target+"\n")#[k]!!!!!!!!
                        f_hypo.write(got+"\n")
                if i==0:
                    for sth in f_ref:
                        sth.close()
                f_hypo.close()
                print("finish one")
                        # if len(ref) > 3 and len(hypothesis) > 3:
                        # if all([len(sen) > 3 for sen in ref]) and len(hypothesis) > 3:
                        #     list_of_refs.append(ref)
                        #     hypotheses.append(hypothesis)
