from matplotlib import pyplot as plt
import numpy as np

N = 4
ind = np.arange(N)
width = 0.15

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

multihead = [0.111355576, 0.323647115, 0.261693091, 0.271005348]
singlehead = [0.108358574, 0.249243201, 0.258018339, 0.248603018]
cogcn = [0.07984712, 0.308077274, 0.257054019, 0.23142632]
node2vec = [0.042856294259662336, 0.05863240289469798, 0.06674660271782573, 0.039627039627039624]

bar1 = plt.bar(ind, multihead, width, color='#154360', edgecolor='black')
bar2 = plt.bar(ind + width, singlehead, width, color='#2874A6', edgecolor='black')
bar3 = plt.bar(ind + width * 2, cogcn, width, color='#ABEBC6')
bar4 = plt.bar(ind + width * 3, node2vec, width, color='#D2B4DE')

# plt.xlabel("Dates")
plt.ylabel('Structural Modularity', **font)
# plt.title("Players Score")

plt.xticks(ind + width, ['Daytrader', 'PBW', 'AcmeAir', 'DietApp'])
plt.legend((bar1, bar2, bar3, bar4), ('Multi-Head GAT', 'Single-Head GAT', 'CO-GCN', 'Node2vec'))
plt.show()