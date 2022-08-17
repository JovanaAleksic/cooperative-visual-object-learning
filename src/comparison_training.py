#This is just printing file, which takes the data and plots it

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


number_of_training_samples_list = [1, 5, 10, 15, 20, 30, 40, 50, 70, 80, 100, 120, 130, 150, 160, 180, 200]

classification_rate_baseline = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate197.pickle", "rb"))[0]
classification_rate_baseline100 = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate196.pickle", "rb"))[0]
classification_rate_baseline10 = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate195.pickle", "rb"))[0]
classification_rate_baseline2 = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate194.pickle", "rb"))[0]
# classification_rate_baseline_new = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate216.pickle", "rb"))[0]
# classification_rate_baseline_bnb_new = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate222.pickle", "rb"))[0]
# classification_rate_training = pickle.load(open("//hri/storage/user/jradojev/Version1/Strategies/Training/Training_Rate16.pickle", "rb"))[0]
# classification_rate_training2 = pickle.load(open("//hri/storage/user/jradojev/Version1/Strategies/Training2/Training_Rate10.pickle", "rb"))[0]
# classification_rate_training3 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/1000D/Training_Rate424.pickle", "rb"))[0]
# classification_rate_training31 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/1000D/Training_Rate425.pickle", "rb"))[0]
# classification_rate_training32 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/1000D/Training_Rate428.pickle", "rb"))[0]
# classification_rate_training33 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3/1000D/Training_Rate431.pickle", "rb"))[0]
# classification_rate_training3_bnb= pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training3+bnb/Training_Rate44.pickle", "rb"))[0]
# classification_rate_training5 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training5/Training_Rate315.pickle", "rb"))[0]
classification_rate_test = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/Test_Rate490.pickle", "rb"))[0]
classification_rate_test3 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/Test_Rate495.pickle", "rb"))[0]
# classification_rate_training61 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training6/Training_Rate548.pickle", "rb"))[0]
classification_rate_training6 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training6/Training_Rate665.pickle", "rb"))[0]
classification_rate_training7 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training7/Training_Rate684.pickle", "rb"))[0]
# classification_rate_training61 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training6/Training_Rate92.pickle", "rb"))[0]
# classification_rate_training51 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training5/Training_Rate47.pickle", "rb"))[0]
# classification_rate_training71 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training7/Training_Rate515.pickle", "rb"))[0]
# classification_rate_test1 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/Test_Rate421.pickle", "rb"))[0]
# classification_rate_test2 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/Test_Rate423.pickle", "rb"))[0]
classification_rate_testtraining7 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/TestTraining7/Training_Rate573.pickle", "rb"))[0]
classification_rate_testtraining3 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/TestTraining3/Training_Rate576.pickle", "rb"))[0]
# classification_rate_training61 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Training6/Training_Rate92.pickle", "rb"))[0]
classification_rate_baseline_bnb = pickle.load(open("/hri/storage/user/jradojev/Version1/Baseline_data/Baseline_Rate235.pickle", "rb"))[0]
# classification_rate_test22 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test2/Test_Rate109.pickle", "rb"))[0]
classification_rate_test3training3 = pickle.load(open("//hri/storage/user/jradojev/Version1/Strategies/Test3Training3/Training_Rate407.pickle", "rb"))[0]
classification_rate_test3training7 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3Training7/Test3Training7_Rate202.pickle", "rb"))[0]
# classification_rate_testc = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test/New/Test_Rate122.pickle", "rb"))[0]
# classification_rate_test2 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test2/Test_Rate136.pickle", "rb"))[0]
# classification_rate_test21 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test2/Test_Rate118.pickle", "rb"))[0]
# classification_rate_test31 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/Test_Rate425.pickle", "rb"))[0]
# classification_rate_test32 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/Test_Rate428.pickle", "rb"))[0]
# classification_rate_test33 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/Test3/Test_Rate431.pickle", "rb"))[0]
# classification_rate_testtraining1 = pickle.load(open("/hri/storage/user/jradojev/Version1/Strategies/TestTraining/TestTraining_Rate8.pickle", "rb"))[0]

fig = plt.figure()
ax = fig.gca()
ax.axis('tight')
fig.tight_layout()
# ax.set_xticks(np.arange(0, 200, 10))
plt.xticks(range(0, 200, 10), fontsize=15)
# ax.set_yticks(np.arange(0, 1., 0.1))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
# plt.plot(number_of_training_samples_list, classification_rate_baseline, '-ko', markersize=12, linewidth=2, label='Baseline 1000D')
# plt.plot(number_of_training_samples_list, classification_rate_baseline100, '-ko', markersize=12, linewidth=2, label='Baseline 100D') #,color='#708090
# plt.plot(number_of_training_samples_list, classification_rate_baseline10, '-ko', markersize=12, linewidth=2, label='Baseline 10D') #, color='#696969'
plt.plot(number_of_training_samples_list, classification_rate_baseline2, '-ko', markersize=12, linewidth=2, label='Baseline 2D') # color='#d3d3d3'
plt.plot(number_of_training_samples_list, classification_rate_baseline_bnb, '-o', color='#bebebe', markersize=12, linewidth=2, label='Check')
# plt.plot(number_of_training_samples_list, classification_rate_baseline_new, '-o', color='#8b7765', markersize=12, linewidth=2, label='Baseline alal novo')
# plt.plot(number_of_training_samples_list, classification_rate_baseline_bnb_new, '-o', color='#708090', markersize=12, linewidth=2, label='Baseline 100al novo')
# plt.plot(number_of_training_samples_list, classification_rate_training, '-bs',  markersize=12, linewidth=2, label='Training simple')
# plt.plot(number_of_training_samples_list, classification_rate_training3, '-bs',  markersize=12, linewidth=2, label='Training strategy 1')
# plt.plot(number_of_training_samples_list, classification_rate_training3_bnb, '-s',color = '#40e0d0',  markersize=12, linewidth=2, label='Training strategy 1+')
# plt.plot(number_of_training_samples_list, classification_rate_training5, '-s', color = '#00ffff',  markersize=12, linewidth=2,  label='Training strategy 2')
plt.plot(number_of_training_samples_list, classification_rate_training6, '-s', color = '#0000ff',  markersize=12, linewidth=2,  label='Training strategy 1')
# plt.plot(number_of_training_samples_list, classification_rate_training61, '-s', color = '#0000ff',  markersize=12, linewidth=2,  label='Training strategy 6')
plt.plot(number_of_training_samples_list, classification_rate_training7, '-s', color='#00bfff', markersize=12,
         linewidth=2, label='Training strategy 2')
# plt.plot(number_of_training_samples_list, classification_rate_training71, '-s', color='#00ffff', markersize=12,
#          linewidth=2, label='Training strategy 7 415')
# plt.plot(number_of_training_samples_list, classification_rate_training61, '-s', color = '#4682b4',  markersize=12, linewidth=2,  label='Training strategy 3 staro')
# plt.plot(number_of_training_samples_list, classification_rate_training51, '-rs', label='Training 51')
plt.plot(number_of_training_samples_list, classification_rate_test, '-g^', markersize=12, linewidth=2,  label='Testing strategy 1') #color='#cd5c5c'
# plt.plot(number_of_training_samples_list, classification_rate_test1, '-^', color='#cd853f', markersize=12, linewidth=2,  label='Testing strategy 1 421')
# plt.plot(number_of_training_samples_list, classification_rate_test2, '-^', color='#a52a2a', markersize=12, linewidth=2,  label='Testing strategy 1 423')
# plt.plot(number_of_training_samples_list, classification_rate_testc, '-k^', markersize=12, linewidth=2,  label='Testing strategy 1 0.0')
# plt.plot(number_of_training_samples_list, classification_rate_test2, '-^', color='#32cd32', markersize=12, linewidth=2, label='Testing strategy 2 136')
# plt.plot(number_of_training_samples_list, classification_rate_test21, '-g^', markersize=12, linewidth=2,  label='Testing strategy 2 118')
# plt.plot(number_of_training_samples_list, classification_rate_test22, '-r^', markersize=12, linewidth=2,  label='Testing strategy 2 109')
plt.plot(number_of_training_samples_list, classification_rate_test3, '-^', color='#32cd32', markersize=12, linewidth=2, label='Testing strategy 2')
# plt.plot(number_of_training_samples_list, classification_rate_test31, '-^', color='#7fffd4', markersize=12, linewidth=2, label='Testing strategy 3 425')
# plt.plot(number_of_training_samples_list, classification_rate_test32, '-^', color='#006400', markersize=12, linewidth=2, label='Testing strategy 3 428')
# plt.plot(number_of_training_samples_list, classification_rate_test33, '-^', color='#7fff00', markersize=12, linewidth=2, label='Testing strategy 3 431')
# plt.plot(number_of_training_samples_list, classification_rate_testtraining, '-rD', markersize=12, linewidth=2,  label='Testing 1 & Training 1 strategy')
# plt.plot(number_of_training_samples_list, classification_rate_testtraining1, '-ro', label='TestTraining')
plt.plot(number_of_training_samples_list, classification_rate_testtraining7, '-rD', markersize=12, linewidth=2, label='Test 1 Training 2')
plt.plot(number_of_training_samples_list, classification_rate_testtraining3, '-D', color='#b22222', markersize=12, linewidth=2, label='Test 1 Training 1')
plt.plot(number_of_training_samples_list, classification_rate_test3training7, '-D', color='#ff8c00', markersize=12, linewidth=2, label='Test2 Training2')
plt.plot(number_of_training_samples_list, classification_rate_test3training3, '-D', color='#ff1493', markersize=12, linewidth=2, label='Test2 Training1')
plt.xlabel('Number of training samples per object', fontsize=20, color='black')
plt.ylabel("Average accuracy of classification", fontsize=20)
# plt.title("Baseline vs simple training improvement averaged over 500 runs")
# plt.title("Baseline with training with automatic tresholds over 100 runs for 10 neurons", fontsize=14)
# plt.title("Test+Training vs baseline with 3 new classes over 500 runs for 1000 neurons")
plt.grid()
plt.ylim([0, 0.2])
plt.legend(loc=1, borderpad=1, labelspacing=1.2, prop={'size': 20})
rcParams['axes.xmargin'] = 1
# plt.savefig("test.png", bbox_inches='tight')
plt.show()

########################################################################################################################
#Baseline 1 is a regular baseline
#Baseline 1+ is a baseline with equal number of images as test
#Baseline 3 is a baseline with new objects
#Baseline 4 is combination of 2 and 3
#Training 1 is training_strategy3
#Training 2 is training_strategy3 with equal number of images
#Training 3 is training_strategy3 with automatic tresholds
#Testing strategy 1 is testing strategy with ratios of kappa/gamma/delta
#Testing strategy 2 is testing strategy 1 applied on the case with new objects
#Testing strategy 3 is a new testing strategy
