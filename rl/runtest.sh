#!/bin/bash
#1
#python3 train.py --mode de2en --save_log de2en_rl_step15999  --gpu 0
#cd ../ZRNMT/
#python3 test_1.py --mode de2en --gpu 0 --load_log ../rl/de2en_rl_step15999 --save_dir result_de2en_rl_step15999
#python3 train.py --mode en2de --save_log en2de_rl_step12999  --gpu 1
#cd ../ZRNMT
#python3 test_1.py --mode en2de --gpu 1 --load_log ../rl/en2de_rl_step12999 --save_dir result_en2de_rl_step12999

#2
#python3 train.py --mode de2en --save_log de2en_rl_x_step9000  --gpu 0
#cd ../ZRNMT/
#python3 test_1.py --mode de2en --gpu 0 --load_log ../rl/de2en_rl_x_step9000 --save_dir result_de2en_rl_x_step9000
#python3 train.py --mode de2en --save_log de2en_rl_x_step7000  --gpu 1
#cd ../ZRNMT/
#python3 test_1.py --mode de2en --gpu 1 --load_log ../rl/de2en_rl_x_step7000 --save_dir result_de2en_rl_x_step7000
#python3 train.py --mode en2de --save_log en2de_rl_x_step3000  --gpu 2
#cd ../ZRNMT
#python3 test_1.py --mode en2de --gpu 2 --load_log ../rl/en2de_rl_x_step3000 --save_dir result_en2de_rl_x_step3000
#python3 train.py --mode en2de --save_log en2de_rl_x_step7000  --gpu 3
#cd ../ZRNMT
#python3 test_1.py --mode en2de --gpu 3 --load_log ../rl/en2de_rl_x_step7000 --save_dir result_en2de_rl_x_step7000

#3
#python3 train.py --mode de2en --save_log de2en_rl_y_step12000  --gpu 2
#cd ../ZRNMT/
#python3 test_1.py --mode de2en --gpu 2 --load_log ../rl/de2en_rl_y_step12000 --save_dir result_de2en_rl_y_step12000
#python3 train.py --mode en2de --save_log en2de_rl_y_step15500  --gpu 3
#cd ../ZRNMT
#python3 test_1.py --mode en2de --gpu 3 --load_log ../rl/en2de_rl_y_step15500 --save_dir result_en2de_rl_y_step15500
pwd
cd ../ZRNMT/
sleep 10
pwd

