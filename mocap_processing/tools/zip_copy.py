import os
from os import path
import argparse

parser = argparse.ArgumentParser(description="Devfair Utilities")
parser.add_argument("--zip_data", action="store_true")
parser.add_argument("--copy_data", action="store_true")
parser.add_argument("--unzip_data", action="store_true")
parser.add_argument("--copy_net", action="store_true")
parser.add_argument("--ext_net", action="store_true")
parser.add_argument("--name", type=str, default="net")
args = parser.parse_args()
print(args)

policy_dirs = [
    "ScaDive/data/learning/mimicpfnn_composite_old3_new1_exp1/",
    "ScaDive/data/learning/mimicpfnn_composite_old4_new1_exp3/",
]

policy_nets = [
    "network300",
    "network5300",
]

log_dir = "log/"

""" Copy Training Logs (done in Local Side) """

# if args.copy_rew:
#     # for d in policy_dirs:
#     #     os.makedirs(d, exist_ok=True)
#     # cmd = "scp -T devfair:\""
#     # for d in policy_dirs:
#     #     cmd += "\"%s/log.txt\" "%d
#     # cmd += "\" %s" % d
#     for d in policy_dirs:
#         os.makedirs(d, exist_ok=True)
#         cmd = "scp -T devfair:\""
#         cmd += "\"%s/log.txt\" "%d
#         cmd += "\" %s" % d
#         os.system(cmd)

# ''' Extract Rewards from Log Files (done in Local Side) '''

# if args.ext_rew:
#     # for f in files:
#     #     log_file = "%s/%s.log"%(log_dir, f)
#     #     rew_file = "%s/rew_%s.txt"%(log_dir, f)
#     #     if path.exists(rew_file):
#     #         print("rm %s"%rew_file)
#     #         os.system("rm %s"%rew_file)
#     #     cmd = "python3 %s/rew.py %s >> %s && subl %s" % (log_dir, log_file, rew_file, rew_file)
#     #     os.system(cmd)
#     for d in policy_dirs:
#         os.system("subl %s/log.txt"%d)

""" Compress data (network, log), which should be done in server side """

if args.zip_data:
    cmd = "tar -cvzf %s.tar.gz " % args.name
    for i in range(len(policy_dirs)):
        cmd += "%s%s " % (policy_dirs[i], policy_nets[i])
        cmd += "%s%s " % (policy_dirs[i], "log.txt")
    os.system(cmd)

""" Copy data (network, log), which should be done in local side """

if args.copy_data:
    os.system("scp -T devfair:~/Research/%s.tar.gz ~/Research" % args.name)

""" Extract data (network, log), which should be done in local side """

if args.unzip_data:
    for d in policy_dirs:
        os.makedirs(d, exist_ok=True)
    cmd = "tar -xvzf %s.tar.gz" % args.name
    os.system(cmd)
    for d in policy_dirs:
        os.system("subl %s/log.txt" % d)
