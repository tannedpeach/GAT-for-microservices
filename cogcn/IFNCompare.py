from matplotlib import pyplot as plt
import numpy as np

N = 4
ind = np.arange(N)
width = 0.15

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

multihead = [22.666666666666664, 9.333333333333332, 4.800000000000001, 2.1]
singlehead = [22.666666666666664, 9.333333333333332, 4.800000000000001, 2.625]
cogcn = [22.666666666666664, 9.333333333333332, 5.333333333333333, 2.333333333333333]
node2vec = [34, 14, 8, 3.5]

bar1 = plt.bar(ind, multihead, width, color='#154360', edgecolor='black')
bar2 = plt.bar(ind + width, singlehead, width, color='#2874A6', edgecolor='black')
bar3 = plt.bar(ind + width * 2, cogcn, width, color='#ABEBC6')
bar4 = plt.bar(ind + width * 3, node2vec, width, color='#D2B4DE')

# plt.xlabel("Dates")
plt.ylabel('IFN', **font)
# plt.title("Players Score")

plt.xticks(ind + width, ['Daytrader', 'PBW', 'AcmeAir', 'DietApp'])
plt.legend((bar1, bar2, bar3, bar4), ('Multi-Head GAT', 'Single-Head GAT', 'CO-GCN', 'Node2vec'))
plt.show()