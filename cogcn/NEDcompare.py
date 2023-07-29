from matplotlib import pyplot as plt
import numpy as np

N = 4
ind = np.arange(N)
width = 0.15

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

multihead = [0.2054794520547945, 0.42857142857142855, 0.38461538461538464, 0.8]
singlehead = [0.136986301369863, 0.5357142857142857, 0.38461538461538464, 0.25]
cogcn = [0.0821917808219178, 0.21428571428571427, 0.15384615384615385, 0.1]
node2vec = [0.1232876712328767, 0.21428571428571427, 0.6538461538461539, 0.75]

bar1 = plt.bar(ind, multihead, width, color='#154360', edgecolor='black')
bar2 = plt.bar(ind + width, singlehead, width, color='#2874A6', edgecolor='black')
bar3 = plt.bar(ind + width * 2, cogcn, width, color='#ABEBC6')
bar4 = plt.bar(ind + width * 3, node2vec, width, color='#D2B4DE')

# plt.xlabel("Dates")
plt.ylabel('NED', **font)
# plt.title("Players Score")


plt.xticks(ind + width, ['Daytrader', 'PBW', 'AcmeAir', 'DietApp'])
plt.legend((bar1, bar2, bar3, bar4), ('Multi-Head GAT', 'Single-Head GAT', 'CO-GCN', 'Node2vec'))



plt.show()