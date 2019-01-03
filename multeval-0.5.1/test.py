import os
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--load_file', type=str, default="../rl/result_en2de_rl_x_test2/en2de_")
parser.add_argument('--number', type=int, default=20)
args = parser.parse_args()
dire = args.load_file
#dire = "../ZRNMT/result_test/de2en_"
language = "de"
command = "./multeval.sh eval --refs {}ref* \
                   --hyps-baseline {}hypo_1 ".format(dire,dire)
n=args.number
for i in range(1,n):
 command+=r" --hyps-sys{} {}hypo_{}".format(i,dire,i+1)
command+=r" --meteor.language {}".format(language)
os.system(command)
# nohup python test.py --load_file ../ZRNMT/result_en2de_rl_step12999/en2de_ > result_en2de_rl_step12999 &
# nohup python test.py --load_file ../ZRNMT/result_en2de_rl_x_step3000/en2de_ > result_en2de_rl_x_step3000 &
# nohup python test.py --load_file ../ZRNMT/result_en2de_rl_x_step7000/en2de_ > result_en2de_rl_x_step7000 &
# nohup python test.py --load_file ../ZRNMT/result_en2de_rl_y_step15500/en2de_ > result_en2de_rl_y_step15500 &
# nohup python test.py --load_file ../ZRNMT/result_de2en_rl_step15999/de2en_ > result_de2en_rl_step15999 &
# nohup python test.py --load_file ../ZRNMT/result_de2en_rl_x_step9000/de2en_ > result_de2en_rl_x_step9000 &
# nohup python test.py --load_file ../ZRNMT/result_de2en_rl_x_step7000/de2en_ > result_de2en_rl_x_step7000 &
# nohup python test.py --load_file ../ZRNMT/result_de2en_rl_y_step12000/de2en_ > result_de2en_rl_y_step12000 &
