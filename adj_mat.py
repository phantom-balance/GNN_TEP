import torch 
import numpy as np 
import os
import pickle

# node arrangement
# xmeas_1,xmeas_2,xmeas_3,xmeas_4,xmeas_5,xmeas_6,xmeas_7,xmeas_8,xmeas_9,xmeas_10,xmeas_11,xmeas_12,xmeas_13,xmeas_14,xmeas_15,xmeas_16,xmeas_17,xmeas_18,xmeas_19,xmeas_20,xmeas_21,xmeas_22,xmeas_23,xmeas_24,xmeas_25,xmeas_26,xmeas_27,xmeas_28,xmeas_29,xmeas_30,xmeas_31,xmeas_32,xmeas_33,xmeas_34,xmeas_35,xmeas_36,xmeas_37,xmeas_38,xmeas_39,xmeas_40,xmeas_41,xmv_1,xmv_2,xmv_3,xmv_4,xmv_5,xmv_6,xmv_7,xmv_8,xmv_9,xmv_10,xmv_11

A = np.zeros([82,82], dtype=int)

A[0,43]=1 # xmeas_1---> xmv_3
A[1,41]=1 # xmeas_2---> xmv_1
A[2,42]=1 # xmeas_3---> xmv_2
A[3,44]=1 # xmeas_4---> xmv_4
A[6,74]=1 # xmeas_7---> cn_7
A[7,81]=1 # xmeas_8---> cn_14
A[8,50]=1 # xmeas_9---> xmv_10
A[9,46]=1 # xmeas_10---> xmv_6
A[10,51]=1 # xmeas_11---> xmv_11
A[11,73]=1 # xmeas_12---> cn_6
A[13,47]=1 # xmeas_14---> xmv_7
A[14,72]=1 # xmeas_15---> cn_5
A[16,48]=1 # xmeas_17---> xmv_8
A[16,68]=1 # xmeas_17---> cn_1
A[22,79]=1 # xmeas_23---> cn_12
A[22,80]=1 # xmeas_23---> cn_13
A[24,79]=1 # xmeas_25---> cn_12
A[24,80]=1 # xmeas_25---> cn_13
A[39,71]=1 # xmeas_40---> cn_4
A[41,58]=1 # xmv_1---> stream_2
A[42,59]=1 # xmv_2---> stream_3
A[43,57]=1 # xmv_3---> stream_1
A[44,60]=1 # xmv_4---> stream_4
A[45,54]=1 # xmv_5---> compressor
A[46,63]=1 # xmv_6---> stream_9
A[47,64]=1 # xmv_7---> stream_10
A[48,65]=1 # xmv_8---> stream_11
A[49,67]=1 # xmv_9---> stream_14
A[50,66]=1 # xmv_10--> stream_12
A[51,53]=1 # xmv_11---> condensor
A[52,6]=1 # reactor---> xmeas_7
A[52,7]=1 # reactor---> xmeas_8
A[52,8]=1 # reactor---> xmeas_9
A[52,53]=1 # reactor---> condensor
A[52,66]=1 # reactor---> stream_12
A[53,21]=1 # condensor---> xmeas_22
A[53,55]=1 # condensor---> separator
A[54,10]=1 # compressor---> xmeas_11
A[54,19]=1 # compressor---> xmeas_20
A[54,62]=1 # compressor---> stream_8
A[55,11]=1 # separator ---> xmeas_12
A[55,12]=1 # separator ---> xmeas_13
A[55,54]=1 # separator ---> compressor
A[55,63]=1 # separator ---> stream_9
A[55,64]=1 # separator ---> stream_10
A[56,14]=1 # stripper ---> xmeas_15
A[56,15]=1 # stripper ---> xmeas_16
A[56,17]=1 # stripper ---> xmeas_18
A[56,61]=1 # stripper ---> stream_6
A[56,65]=1 # stripper ---> stream_11
A[57,0]=1 # stream_1 ---> xmeas_1
A[57,61]=1 # stream_1 ---> stream_6
A[58,1]=1 # stream_2 ---> xmeas_2
A[58,61]=1 # stream_2 ---> stream_6
A[59,2]=1 # stream_3 ---> xmeas_3
A[59,61]=1 # stream_3 ---> stream_6
A[60,3]=1 # stream_4 ---> xmeas_4
A[60,56]=1 # stream_4 ---> stripper
A[61,5]=1 # stream_6 ---> xmeas_6
A[61,22]=1 # stream_6 ---> xmeas_23
A[61,23]=1 # stream_6 ---> xmeas_24
A[61,24]=1 # stream_6 ---> xmeas_25
A[61,25]=1 # stream_6 ---> xmeas_26
A[61,26]=1 # stream_6 ---> xmeas_27
A[61,27]=1 # stream_6 ---> xmeas_28
A[61,52]=1 # stream_6 ---> reactor
A[62,4]=1 # stream_8 ---> xmeas_5
A[62,61]=1 # stream_8 ---> stream_6
A[63,9]=1 # stream_9 ---> xmeas_10
A[63,28]=1 # stream_9 ---> xmeas_29
A[63,29]=1 # stream_9 ---> xmeas_30
A[63,30]=1 # stream_9 ---> xmeas_31
A[63,31]=1 # stream_9 ---> xmeas_32
A[63,32]=1 # stream_9 ---> xmeas_33
A[63,33]=1 # stream_9 ---> xmeas_34
A[63,34]=1 # stream_9 ---> xmeas_35
A[63,35]=1 # stream_9 ---> xmeas_36
A[64,13]=1 # stream_10 ---> xmeas_14
A[64,56]=1 # stream_10 ---> stripper
A[65,16]=1 # stream_11 ----> xmeas_17
A[65,36]=1 # stream_11 ----> xmeas_37
A[65,37]=1 # stream_11 ----> xmeas_38
A[65,38]=1 # stream_11 ----> xmeas_39
A[65,39]=1 # stream_11 ----> xmeas_40
A[65,40]=1 # stream_11 ----> xmeas_41
A[66,20]=1 # stream_12 ---> xmeas_21
A[67,18]=1  # stream_14 ---> xmeas_19
A[67,56]=1 # stream_14 ---> stripper
A[68,41]=1 # cn_1 ---> xmv_1
A[68,42]=1 # cn_1 ---> xmv_2
A[68,43]=1 # cn_1 ---> xmv_3
A[68,44]=1 # cn_1 ---> xmv_4
A[68,46]=1 # cn_1 ---> xmv_6
A[68,47]=1 # cn_1 ---> xmv_7
A[68,48]=1 # cn_1 ---> xmv_8
A[68,69]=1 # cn_1 ---> cn_2
A[68,70]=1 # cn_1 ---> cn_3
A[69,42]=1 # cn_2 ---> xmv_2
A[70,41]=1 # cn_3 ---> xmv_1
A[71,69]=1 # cn_4 ---> cn_2
A[71,70]=1 # cn_4 ---> cn_3
A[72,48]=1 # cn_5 ---> xmv_8
A[73,47]=1 # cn_6 ---> xmv_7
A[74,46]=1 # cn_7 ---> xmv_6
A[75,44]=1 # cn_8 ---> xmv_4
A[76,43]=1 # cn_9 ---> xmv_3
A[77,75]=1 # cn_10 ---> cn_8
A[77,76]=1 # cn_10 ---> cn_9
A[78,75]=1 # cn_11 ---> cn_8
A[78,76]=1 # cn_11 ---> cn_9
A[79,77]=1 # cn_12 ---> cn_10
A[80,78]=1 # cn_13 ---> cn_11
A[81,51]=1 # cn_14 ---> xmv_11

print("No. of edges:",np.count_nonzero(A==1))

with open(f'processed_data/directed_adjacency_matrix.p', 'wb') as f:
    pickle.dump(A, f)

A_T= np.transpose(A,axes=None)
A_ud = A+A_T

with open(f'processed_data/undirected_adjacency_matrix.p', 'wb') as f:
    pickle.dump(A_ud, f)

print("No. of edges:",np.count_nonzero(A_ud==1))

D = pickle.load(open("processed_data/directed_adjacency_matrix.p", "rb"))
U = pickle.load(open("processed_data/undirected_adjacency_matrix.p", "rb"))

A = []
B = []

for i in range(len(D[0])):
  for j in range(len(D[0])):
    if (D[i][j]!=0):
      A.append(i)
      B.append(j)
    
# print(A)
# print(B)
A=torch.tensor(A)
B=torch.tensor(B)
# print(A.shape)
# print(B.shape)
D_edge_list = torch.stack((A,B),dim=0)
print("directed_list_shape:",D_edge_list.shape)

with open(f'processed_data/directed_adjacency_list.p', 'wb') as f:
    pickle.dump(D_edge_list, f)

A = []
B = []

for i in range(len(U[0])):
  for j in range(len(U[0])):
    if (U[i][j]!=0):
      A.append(i)
      B.append(j)
    
# print(A)
# print(B)
A=torch.tensor(A)
B=torch.tensor(B)
# print(A.shape)
# print(B.shape)
U_edge_list = torch.stack((A,B),dim=0)
print("undirected_list_shape:",U_edge_list.shape)

with open(f'processed_data/undirected_adjacency_list.p', 'wb') as f:
    pickle.dump(U_edge_list, f)