import os
from data_load import *
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import tensorflow as tf
from hyperparams_rl import Hyperparams as hp
import numpy as np
import math,os,argparse
from train import Graph#!!!
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def concat(name):
    new_name = name+"1"
    os.system(r"sed -r 's/(@@ )|(@@ ?$)//g' < {} > {} ".format(name, new_name))
    os.system("rm {}".format(name))
    os.system("mv {} {}".format(new_name, name))

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

if __name__ == '__main__':
    #argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default="de2en")
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--task', type=str, default="task2")
    parser.add_argument('--save_dir', type=str, default="result_test")
    parser.add_argument('--load_log', type=str, default="logdir")
    args = parser.parse_args()
    mode = args.mode
    task = args.task
    load_log = args.load_log
    save_dir = args.save_dir
    exam_set = "val"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    #setting
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    translator = eval("idx2{}".format(mode[-2:]))
    save_list = os.listdir(load_log)
    file_list = set(["/"+save.split('.')[0] for save in save_list if save!="checkpoint" and save!="sample"])
    print(file_list)
    file_list = sorted(file_list,key=lambda x:int(x.split('_')[1]))#不同位置不一样
    #自定义
    # file_list = ["/step_12500"]
    #load graph
    g = Graph(is_training=False, beam_width=5, mode=mode)
    print("Graph loaded")
    # Load vocabulary
    x_val,Targets_val,idents = load_test_rl_data(set=exam_set,task=task,language=mode[:2])
    num_batch_val = int(math.ceil(len(x_val) / hp.batch_size_test))
    if exam_set=="val":
        total_num = 1014
    else:
        total_num = 1000
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
    #start
    save_num = 1
    saver = tf.train.Saver(var_list=g.value_list)
    with g.graph.as_default():
        for k,file in enumerate(file_list):
            with tf.Session() as sess:
                #restore
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,load_log+file)#mode
                print("restore")
                f_hypo = open("{}/{}_hypo_{}".format(save_dir, mode, save_num), "w+")
                # f_hypo_all = open("{}/{}_hypo_all_{}".format(save_dir, mode, save_num), "w+")
                save_num += 1
                prob_list,preds_list = [],[]
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
                    preds = process(preds)*preds
                    feed_dict = {
                        g.x: x_val[j * hp.batch_size_test:(j + 1) * hp.batch_size_test],
                        g.dropout_rate_tran: 0.0,
                        g.is_inference: True,
                        g.y:preds
                    }
                    prob = sess.run(g.prob, feed_dict)
                    prob_list.extend(prob)
                    preds = np.concatenate((preds[:, 1:], np.zeros((len(targets), 1))), axis=1)
                    # corporate
                    for pred in preds:
                        got = " ".join(translator[idx] for idx in pred).split("</S>")[0].strip()
                        preds_list.append(got)
                #choose the max prob
                prob_list = np.array(prob_list)
                preds_best,Targets_best = [],[]
                for i in range(total_num):
                    index = [j for j in range(len(idents)) if idents[j]==i]
                    #a = np.argmax([sentence_bleu(Targets_val[j],preds_list[j]) for j in index])
                    a = np.argmax(prob_list[index])
                    b = index[a]
                    preds_best.append(preds_list[b])
                    Targets_best.append(Targets_val[b])
                #write ref file
                if k==0:
                    for i in range(len(Targets_best)):
                        for k, l in enumerate(f_ref):
                            if task == "task2":
                                l.write(Targets_best[i][k] + "\n")
                            else:
                                l.write(Targets_best[i] + "\n")
                    for sth in f_ref:
                        sth.close()
                        concat(sth.name)
                #write target file
                for i in range(len(preds_best)):
                    f_hypo.write(preds_best[i]+"\n")
                f_hypo.close()
                concat(f_hypo.name)
                # for i in range(total_num):
                #     f_hypo_all.write(str(i) + "\n")
                #     index = [j for j in range(len(idents)) if idents[j] == i]
                #     for item in index:
                #         f_hypo_all.write(preds_list[item])
                #         f_hypo_all.write(" "+str(prob_list[item])+"\n")
                # f_hypo_all.close()
                print("finish one")
