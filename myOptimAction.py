import numpy as np
import pandas as pd


def ComeFrom(maxprice, pricelist):
	parent = 100
	for i in range(len(pricelist)):
		if maxprice == pricelist[i]:
			if i == 0:
				parent = -1
			else:
				parent = i
	return parent


def myOptimAction(priceMat, transFeeRate):
	priceMat = pd.DataFrame(priceMat)
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
	Price_table = np.vstack((P0,P1,P2,P3,P4))
	DP_table[0][0] = initial_cash
	DP_action = np.zeros((5,len(P1)))

	DP_action[0][0] = -1
	DP_action[1][0] = -1
	DP_action[2][0] = -1
	DP_action[3][0] = -1
	DP_action[4][0] = -1
	DP_action = np.zeros((5,len(P1)))


	#print(DP_table.shape, DP_action.shape)

	DP_table[1][0] = DP_table[0][0]/Price_table[1][0]
	DP_table[2][0] = DP_table[0][0]/Price_table[2][0]
	DP_table[3][0] = DP_table[0][0]/Price_table[3][0]
	DP_table[4][0] = DP_table[0][0]/Price_table[4][0]
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
		DP_action[2][i] = ComeFrom(DP_table[2][i], stock2_candidate)

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
	#print(DP_table)
	DP_action = pd.DataFrame(DP_action)
	#print(DP_action)
	final_reward = [DP_table.iloc[0,-1]/DP_table.iloc[0,0], DP_table.iloc[1,-1]*Price_table.iloc[1,-1]/DP_table.iloc[0,0],
		DP_table.iloc[2,-1]*Price_table.iloc[2,-1]/DP_table.iloc[0,0],
		DP_table.iloc[3,-1]*Price_table.iloc[3,-1]/DP_table.iloc[0,0],
		DP_table.iloc[4,-1]*Price_table.iloc[4,-1]/DP_table.iloc[0,0]]

	best_reward = max(final_reward)
	#print(ComeFrom(best_reward, final_reward))
	#print(best_reward)
		
	trace = np.zeros(len(P1)+1)
	trace[-2] = DP_action.iloc[0,-1]

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
	#print(trace)
	actionMat = []
	#time.sleep(10)

	for i in range(0,len(P1)):
		if(trace[i]==trace[i+1]):
			continue
		day = i+1
		a = int(trace[i])
		b = int(trace[i+1])
		if(a == -1):
			z = DP_table.iloc[0,i]
		else:
			z = Price_table.iloc[int(a)+1,i]*DP_table.iloc[int(a)+1,i]
		#else:
		#	z = 
		action = [day, a, b, z]
		actionMat.append(action)
		#print(action)
		#exit()
	return actionMat



