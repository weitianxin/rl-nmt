import argparse,os
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--mode', type=str, default=".")
args = parser.parse_args()
mode = args.mode
print(os.listdir(mode))
