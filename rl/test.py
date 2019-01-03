import os
from data_load import *
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import tensorflow as tf
from hyperparams_rl import Hyperparams as hp
import numpy as np
import math,os,argparse
from train import Graph#!!!

def concat(name):
    new_name = name+"1"
    os.system(r"sed -r 's/(@@ )|(@@ ?$)//g' < {} > {} ".format(name, new_name))
    os.system("rm {}".format(name))
    os.system("mv {} {}".format(new_name, name))

if __name__ == '__main__':
    #argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default="de2en")
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--save_dir', type=str, default="result")
    parser.add_argument('--task', type=str, default="task2")
    parser.add_argument('--load_log', type=str, default="de2en_rl_1")
    args = parser.parse_args()
    mode = args.mode
    task = args.task
    save_dir = args.save_dir
    load_log = args.load_log
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #setting
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    translator = eval("idx2{}".format(mode[-2:]))
    save_list = os.listdir(load_log)#mode
    file_list = set(["/"+save.split('.')[0] for save in save_list if save!="checkpoint"])
    file_list = sorted(file_list,key=lambda x:int(x.split('_')[1]))#不同位置不一样
    #load graph
    g = Graph(is_training=False, beam_width=5, mode=mode)
    print("Graph loaded")
    # Load vocabulary
    x_val,Targets_val = load_test_rl_data(set="val",task=task,language=mode[:2])
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
    save_num = 1
    saver = tf.train.Saver(var_list=g.value_list)
    with g.graph.as_default():
        for i,file in enumerate(file_list):
            with tf.Session() as sess:
                #restore
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,load_log+file)#mode
                print("restore")
                f_hypo = open("{}/{}_hypo_{}".format(save_dir, mode, save_num), "w+")
                save_num += 1
                for j in range(num_batch_val):
                    # bleu score
                    # cal_pred
                    targets = Targets_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test]
                    feed_dict = {
                        g.x: x_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test],
                        g.dropout_rate_tran: 0.0,
                        g.is_inference: True
                    }
                    preds = sess.run(g.preds, feed_dict)
                    preds = np.concatenate((preds[:, 1:], np.zeros((len(targets), 1))), axis=1)
                    # corporate
                    for pred in preds:  # sentence-wisex_val_batch
                        got = " ".join(translator[idx] for idx in pred).split("</S>")[0].strip()
                        f_hypo.write(got + "\n")
                f_hypo.close()
                concat(f_hypo.name)
                print("finish one")
