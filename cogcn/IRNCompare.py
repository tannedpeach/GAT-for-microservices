from matplotlib import pyplot as plt
import numpy as np

N = 4
ind = np.arange(N)
width = 0.15

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

multihead = [2, 4, 20, 2]
singlehead = [2, 20, 36, 6]
cogcn = [198, 76, 44, 12]
node2vec = [212, 134, 76, 36]

bar1 = plt.bar(ind, multihead, width, color='#154360', edgecolor='black')
bar2 = plt.bar(ind + width, singlehead, width, color='#2874A6', edgecolor='black')
bar3 = plt.bar(ind + width * 2, cogcn, width, color='#ABEBC6')
bar4 = plt.bar(ind + width * 3, node2vec, width, color='#D2B4DE')

# plt.xlabel("Dates")
plt.ylabel('IRN', **font)
# plt.title("Players Score")

plt.xticks(ind + width, ['Daytrader', 'PBW', 'AcmeAir', 'DietApp'])
plt.legend((bar1, bar2, bar3, bar4), ('Multi-Head GAT', 'Single-Head GAT', 'CO-GCN', 'Node2vec'))
plt.show()