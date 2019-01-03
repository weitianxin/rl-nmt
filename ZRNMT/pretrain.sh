#预训练
nohup python3 train.py --mode de2en --save_log de2en_pre --save_dir result_de2en --gpu 1 &
nohup python3 train.py --mode en2de --save_log en2de_pre --save_dir result_en2de --gpu 2 &
nohup python3 train_y.py --mode en2de --save_log en2de_pre_y_1 --save_dir result_en2de_y_1 --gpu 2 --save_file blue_en2de_ypre &
nohup python3 train_y.py --mode de2en --save_log de2en_pre_y_1 --save_dir result_de2en_y_1 --gpu 3 --save_file blue_de2en_ypre &

#rl训练 x指在encoder预训练的方式，y在decoder预训练的方式
nohup python3 train_y.py --mode en2de --save_log en2de_rl_y --save_dir result_en2de_rl_y --gpu 2 --save_file blue_en2de_y &
nohup python3 train_y.py --mode de2en --save_log de2en_rl_y --save_dir result_de2en_rl_y --gpu 3 --save_file blue_de2en_y &

nohup python3 sample_1.py --mode en2de --save_log en2de_rl_y --save_dir result_en2de_rl_y --gpu 0 &
nohup python3 sample_1.py --mode de2en --save_log de2en_rl_y --save_dir result_de2en_rl_y --gpu 1 &

nohup python3 train.py --mode en2de --save_log en2de_rl_x --save_dir result_en2de_rl_x --gpu 2 --save_file bleu_en2de_rl_x1 &

nohup python3 train.py --mode de2en --save_log de2en_rl_x2 --save_dir result_de2en_rl_x2 --gpu 2 --save_file bleu_de2en_rl_xchange2 &
nohup python3 train.py --mode de2en --save_log de2en_rl_x3 --save_dir result_de2en_rl_x3 --gpu 3 --save_file bleu_de2en_rl_xchange3 &
nohup python3 train.py --mode de2en --save_log de2en_rl_x4 --save_dir result_de2en_rl_x4 --gpu 1 --save_file bleu_de2en_rl_xchange4 &
nohup python3 train.py --mode de2en --save_log de2en_rl_x5 --save_dir result_de2en_rl_x5 --gpu 3 --save_file bleu_de2en_rl_xchange5 &

nohup python3 train.py --mode de2en --save_log de2en_rl_x1 --save_dir result_de2en_rl_x1 --gpu 1 --save_file bleu_de2en_rl_x1 &
nohup python3 train.py --mode en2de --save_log en2de_rl_y1 --save_dir result_en2de_rl_y1 --gpu 2 --save_file bleu_en2de_rl_y1 &
nohup python3 train.py --mode de2en --save_log de2en_rl_y --save_dir result_de2en_rl_y --gpu 3 --save_file bleu_de2en_rl_y &

nohup python3 train.py --mode en2de --save_log en2de_rl_x --save_dir result_en2de_rl_x --gpu 2 --save_file bleu_en2de_rl_x &
nohup python3 train.py --mode de2en --save_log de2en_rl_x --save_dir result_de2en_rl_x --gpu 3 --save_file bleu_de2en_rl_x &

#第二种test方法,
nohup python3 test_1.py --mode de2en --gpu 3 --load_log ../rl/de2en_rl_x --save_dir ../rl/result_de2en_rl_x_test2 &
nohup python3 test_1.py --mode en2de --gpu 1 --load_log ../rl/en2de_rl_x --save_dir ../rl/result_en2de_rl_x_test2 &

nohup python3 test_1.py --mode en2de --gpu 2 --load_log ../rl/en2de_rl_y --save_dir ../rl/result_en2de_rl_y_test2 &
nohup python3 test_1.py --mode de2en --gpu 3 --load_log ../rl/de2en_rl_y --save_dir ../rl/result_de2en_rl_y_test2 &

nohup python3 test_1.py --mode en2de --gpu 0 --load_log en2de_rl_y --save_dir result_en2de_rl_y_test2 &
nohup python3 test_1.py --mode de2en --gpu 1 --load_log de2en_rl_y --save_dir result_de2en_rl_y_test2 &

nohup python3 test_1.py --mode en2de --gpu 2 --load_log en2de_rl_x --save_dir result_en2de_rl_x_test2 &
nohup python3 test_1.py --mode de2en --gpu 3 --load_log de2en_rl_x --save_dir result_de2en_rl_x_test2 &
