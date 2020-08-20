import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
priceMat = pd.read_csv('priceMat.txt', delimiter = ' ', header = None)
ans = pd.read_csv('trace.csv')

P0 = []
P1 = []
P2 = []
P3 = []
P4 = []


for i in range(len(priceMat)):
	P1.append(priceMat.iloc[i,0])
	P2.append(priceMat.iloc[i,1])
	P3.append(priceMat.iloc[i,2])
	P4.append(priceMat.iloc[i,3])
P1.pop(0)
P2.pop(0)
P3.pop(0)
P4.pop(0)
for i in range(len(P1)):
	P0.append(0)

len_of_days = len(P1)
stock_num = 4
initial_cash = 1000
fee = 0.01
DP_table = np.zeros((stock_num+1,len_of_days))
#print(DP_table.shape)
#P1 = [1, 2, 3, 4, 6, 5, 3, 2, 3, 4]
#P2 = [6, 5, 3, 6, 5, 4, 2, 7, 2, 3]
#P3 = [3, 8, 2, 6, 3, 1, 9, 6, 2, 5]
#P4 = [4, 2, 3, 7, 2, 8, 2, 3, 5, 1]

Price_table = np.vstack((P0,P1,P2,P3,P4))
DP_table[0][0] = initial_cash
DP_action = np.zeros((5,len(P1)))
#DP_action[0][0] = -1
#DP_action[1][0] = -1
#DP_action[2][0] = -1
#DP_action[3][0] = -1
#DP_action[4][0] = -1
DP_table[1][0] = DP_table[0][0]/Price_table[1][0]
DP_table[2][0] = DP_table[0][0]/Price_table[2][0]
DP_table[3][0] = DP_table[0][0]/Price_table[3][0]
DP_table[4][0] = DP_table[0][0]/Price_table[4][0]
#DP_action = -1 means it come from cash
#DP_action = 1 means it come from stack1
#DP_action = 2 means it come from stack2
#DP_action = 3 means it come from stack3
#DP_action = 4 means it come from stack4
#print(DP_action.shape)

def ComeFrom(maxprice, pricelist):
	parent = 100
	#print(pricelist)
	#print(maxprice)
	for i in range(len(pricelist)):
		if maxprice == pricelist[i]:
			if i == 0:
				parent = -1
			else:
				parent = i
	return parent



for i in range(1, len_of_days):
	#cash
	cash_candidate = [DP_table[0][i-1], DP_table[1][i-1]*Price_table[1][i]*(1-fee),
							DP_table[2][i-1]*Price_table[2][i]*(1-fee), DP_table[3][i-1]*Price_table[3][i]*(1-fee),
							DP_table[4][i-1]*Price_table[4][i]*(1-fee)]
	DP_table[0][i] = max(cash_candidate)
	DP_action[0][i] = ComeFrom(DP_table[0][i],cash_candidate)


	#stock 1
	stock1_candidate = [DP_table[0][i-1]*(1-fee)/Price_table[1][i], DP_table[1][i-1],(DP_table[2][i-1]*Price_table[2][i]*(1-fee)**2)/Price_table[1][i],
						(DP_table[3][i-1]*Price_table[3][i]*(1-fee)**2)/Price_table[1][i], (DP_table[4][i-1]*Price_table[4][i]*(1-fee)**2)/Price_table[1][i]]

	DP_table[1][i] = max(stock1_candidate)
	DP_action[1][i] = ComeFrom(DP_table[1][i], stock1_candidate)

	#stock2
	stock2_candidate = [DP_table[0][i-1]*(1-fee)/Price_table[2][i],(DP_table[1][i-1]*Price_table[1][i]*(1-fee)**2)/Price_table[2][i] ,DP_table[2][i-1],
						(DP_table[3][i-1]*Price_table[3][i]*(1-fee)**2)/Price_table[2][i], (DP_table[4][i-1]*Price_table[4][i]*(1-fee)**2)/Price_table[2][i]]
	DP_table[2][i] = max(stock2_candidate)
	#print('now')
	DP_action[2][i] = ComeFrom(DP_table[2][i], stock2_candidate)
	#print(stock1_candidate)
	#print(max(stock2_candidate))
	#print(DP_action[2][i])
	#exit()
	#stock3
	stock3_candidate = [DP_table[0][i-1]*(1-fee)/Price_table[3][i],(DP_table[1][i-1]*Price_table[1][i]*(1-fee)**2)/Price_table[3][i],
						(DP_table[2][i-1]*Price_table[2][i]*(1-fee)**2)/Price_table[3][i],DP_table[3][i-1], (DP_table[4][i-1]*Price_table[4][i]*(1-fee)**2)/Price_table[3][i]]
	DP_table[3][i] = max(stock3_candidate)
	DP_action[3][i] = ComeFrom(DP_table[3][i], stock3_candidate)

	#stock4
	stock4_candidate = [DP_table[0][i-1]*(1-fee)/Price_table[4][i], (DP_table[1][i-1]*Price_table[1][i]*(1-fee)**2)/Price_table[4][i],
						(DP_table[2][i-1]*Price_table[2][i]*(1-fee)**2)/Price_table[4][i], (DP_table[3][i-1]*Price_table[3][i]*(1-fee)**2)/Price_table[4][i], DP_table[4][i-1]]
	DP_table[4][i] = max(stock4_candidate)
	DP_action[4][i] = ComeFrom(DP_table[4][i], stock4_candidate)
	
Price_table = pd.DataFrame(Price_table)

DP_table = pd.DataFrame(DP_table)
print(DP_table)
DP_action = pd.DataFrame(DP_action)
print(DP_action)
print(ans)
'''
print(DP_table.iloc[0,-1]/DP_table.iloc[0,0],DP_table.iloc[1,-1]*DP_table.iloc[1+4,-1]/DP_table.iloc[0,0],
	DP_table.iloc[2,-1]*DP_table.iloc[2+4,-1]/DP_table.iloc[0,0],
	DP_table.iloc[3,-1]*DP_table.iloc[3+4,-1]/DP_table.iloc[0,0],
	DP_table.iloc[4,-1]*DP_table.iloc[4+4,-1]/DP_table.iloc[0,0])
'''
final_reward = [DP_table.iloc[0,-1]/DP_table.iloc[0,0], DP_table.iloc[1,-1]*Price_table.iloc[1,-1]/DP_table.iloc[0,0],
		DP_table.iloc[2,-1]*Price_table.iloc[2,-1]/DP_table.iloc[0,0],
		DP_table.iloc[3,-1]*Price_table.iloc[3,-1]/DP_table.iloc[0,0],
		DP_table.iloc[4,-1]*Price_table.iloc[4,-1]/DP_table.iloc[0,0]]

best_reward = max(final_reward)
#print(ComeFrom(best_reward, final_reward))

#print(best_reward)

trace = np.zeros(len(P1))
trace[-2] = DP_action.iloc[0,-1]

#print(trace[-1])
#for i in range(len(P1)-3,-1,-1):
#	trace[i] = DP_action.iloc[int(trace[i+1]),i]
for i in range(len(P1)-3,-1,-1):
	if(int(trace[i+1])==-1):
		trace[i] = DP_action.iloc[0,i]
	else:
		trace[i] = DP_action.iloc[int(trace[i+1]),i]
trace[-1] = -1
	#print(trace)
for i in range(len(P1)):
	if(trace[i]>-1):
		trace[i]=trace[i]-1
'''
for i in range(len(trace)):
	print(trace[i])
'''
#print(trace)
DP_table.to_csv('DP.csv')
DP_action.to_csv('Action.csv')
'''
with open('DP.txt',"w") as outfile:
	for i in range(DP_table.shape[0]):
		#for j in range(DP_table.shape[1]):
		for j in range(10):
			outfile.write(str(DP_table.iloc[i,j]))
			outfile.write(' ')
		outfile.write('\n')

with open('Action.txt',"w")as outfile:
	for i in range(DP_action.shape[0]):
		for j in range(10):
			outfile.write(str(DP_action.iloc[i,j]))
			outfile.write(' ')
		outfile.write('\n')
'''
with open('trace.txt',"w")as outfile:
	for i in range(10):
		outfile.write(str(trace[i]))
		outfile.write(' ')









