import os
from data_load import *
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import tensorflow as tf
from hyperparams import Hyperparams as hp
import numpy as np
import math,os
from train_cap_de import Graph#!!!
mode = "de"#!!!
os.environ["CUDA_VISIBLE_DEVICES"]="1"#！！！
image_path = "../../image/flickr30k_ResNets50_blck4_{}.fp16.npy"
if __name__ == '__main__':
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    test_set = "val"
    translator = idx2de#!!!!!!!!

    save_list = os.listdir(eval("hp.logdir_cap_{}".format(mode)))
    file_list = set(["/"+save.split('.')[0] for save in save_list if save!="checkpoint"])
    file_list = sorted(file_list,key=lambda x:int(x.split('_')[2]))
    g = Graph(False)
    print("Graph loaded")

    # Load vocabulary
    Image_index_val, Targets = load_test_cap_data(set=test_set, language=mode)
    val_images = np.load(image_path.format(test_set))
    num_batch_val = int(math.ceil(len(Targets) / hp.batch_size_test))
    saver = tf.train.Saver(var_list=g.value_list)
    f_ref1 = open("../result/caption/{}/ref_1".format(mode), "a+")
    f_ref2 = open("../result/caption/{}/ref_2".format(mode), "a+")
    f_ref3 = open("../result/caption/{}/ref_3".format(mode), "a+")
    f_ref4 = open("../result/caption/{}/ref_4".format(mode), "a+")
    f_ref5 = open("../result/caption/{}/ref_5".format(mode), "a+")
    f_ref = [f_ref1,f_ref2,f_ref3,f_ref4,f_ref5]
    with g.graph.as_default():
        for i,file in enumerate(file_list):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,eval("hp.logdir_cap_{}".format(mode))+file)
                f_hypo = open("../result/caption/{}/hypo_{}".format(mode, i + 1), "a+")
                print("restore")
                for j in range(num_batch_val):
                    # bleu score
                    # cal_pred
                    targets = Targets[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]
                    image_val = val_images[Image_index_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]]
                    feed_dict_val = {
                        g.image: image_val,
                        g.dropout_rate: 0.0,
                        g.lstm_drop_rate: 0.0
                    }
                    preds = sess.run(g.preds, feed_dict_val)
                    preds = np.concatenate((preds[:, 1:], np.zeros((len(targets), 1))), axis=1)
                    # corporate
                    for target, pred in zip(targets, preds):  # sentence-wise
                        got = " ".join(translator[idx] for idx in pred).split("</S>")[0].strip()
                        if i==0:
                            for k,l in enumerate(f_ref):
                                l.write(target[k]+"\n")
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
