import subprocess

cmd1 = 'python /home/jradojev/Version1/Interaction/Offline/nostrategy2.py'
cmd2 = 'python /home/jradojev/Version1/Interaction/Offline/training_strategy.py'   #simple random strategy / works good
cmd4 = 'python /home/jradojev/Version1/Interaction/Offline/training_strategy2.py'
cmd5 = 'python /home/jradojev/Version1/Interaction/Offline/training_strategy3.py'
cmd6 = 'python /home/jradojev/Version1/Interaction/Offline/training_strategy4.py'
cmd3 = 'python /home/jradojev/Version1/Interaction/Offline/conf_test_strategy.py'
cmd7 = 'python /home/jradojev/Version1/Interaction/Offline/test_training3_1.py'

p1 = subprocess.call(cmd1, shell=True)
# p2 = subprocess.call(cmd2, shell=True)
p3 = subprocess.call(cmd3, shell=True)
# p4 = subprocess.call(cmd4, shell=True)
p5 = subprocess.call(cmd5, shell=True)
# p6 = subprocess.call(cmd6, shell=True)
p7 = subprocess.call(cmd7, shell = True)