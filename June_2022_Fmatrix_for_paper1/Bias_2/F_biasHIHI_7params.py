import numpy as np
import healpy as hp
import sys
from numpy.linalg import inv


u = np.zeros((7,15))

#lmax = 512*3
lmax = 15

   # delta[0] = h0[1]*0.01
   # delta[1] = omegab[0]*0.01
   # delta[2] = omegac[0]*0.01
   # delta[3] = amp_s[0]*0.05
   # delta[4] = opt_dept[0]*0.01
   # delta[5] = sp_index[0]*0.01
for i in range(7):
    u[i] = hp.read_cl('u_grad/wmap_'+str(i)+'_HIHI_zi'+sys.argv[1]+'.fits') 
C_HIHI = np.zeros((15,15))

#Ck9 = np.zeros((15,15))

C_HIHI += np.loadtxt('../HIHI_cov/HIHI_covzHI'+sys.argv[1]+'_av.mat') # 15 bins 35 realisation 

#Ck9 += np.loadtxt('../matrix/kHI0.23_cov_av.mat')



C_HIHI_inv = inv(C_HIHI)
#C9_9_inv = inv(C9_9)
#C13_9_inv = inv(C13_9)
#Ck9_inv = inv(Ck9)

u_HIHI = np.zeros((7,15,1))
#u_HI9_9 = np.zeros((5,15,1))
#u_HI13_9 = np.zeros((5,15,1))
#u_k9 = np.zeros((5,15,1))

for i in range(7):
    u_HIHI[i] = u[i][:lmax].reshape((lmax,1))
    #u_HI9_9[i] = u[i][1][:lmax].reshape((lmax,1))
    #u_HI13_9[i] = u[i][2][:lmax].reshape((lmax,1))
    #u_k9[i] = u[i][3][:lmax].reshape((lmax,1))




'''

u_0_kk = u_0[0][:lmax].reshape((lmax,1))
u_0_cc = u_0[1][:lmax].reshape((lmax,1))
u_0_kc = u_0[2][:lmax].reshape((lmax,1))

u_2_kk = u_2[0][:lmax].reshape((lmax,1))
u_2_cc = u_2[1][:lmax].reshape((lmax,1))
u_2_kc = u_2[2][:lmax].reshape((lmax,1))

u_3_kk = u_3[0][:lmax].reshape((lmax,1))
u_3_cc = u_3[1][:lmax].reshape((lmax,1))
u_3_kc = u_3[2][:lmax].reshape((lmax,1))

u_5_kk = u_5[0][:lmax].reshape((lmax,1))
u_5_cc = u_5[1][:lmax].reshape((lmax,1))
u_5_kc = u_5[2][:lmax].reshape((lmax,1))
'''

F = np.zeros((7,7))
F_HIHI = np.zeros((7,7))
#F_HI9_9 = np.zeros((5,5))
#F_HI13_9 = np.zeros((5,5))
#F_k9 = np.zeros((5,5))
#======================
for i in range (7):
    for j in range (7):
        F_HIHI[i][j] = np.matmul(u_HIHI[i].transpose(),np.matmul(C_HIHI_inv,u_HIHI[j]))[0][0]
        #F_HI9_9[i][j] = np.matmul(u_HI9_9[i].transpose(),np.matmul(C9_9_inv,u_HI9_9[j]))[0][0]
        #F_HI13_9[i][j] = np.matmul(u_HI13_9[i].transpose(),np.matmul(C13_9_inv,u_HI13_9[j]))[0][0]
        #F_k9[i][j] = np.matmul(u_k9[i].transpose(),np.matmul(Ck9_inv,u_k9[j]))[0][0]

F = F_HIHI

#print F

#print(i)
#print (np.matmul(F_HIHI,inv(F_HIHI)))

'''
np.savetxt('F_ij_p18_4params.mat',F)

np.savetxt('F_p18_4params_inv.mat',inv(F))

np.savetxt('F_kk_p18_4params.mat',F_kk)

np.savetxt('F_kk_p18_4params_inv.mat',inv(F_kk))

np.savetxt('F_cc_p18_4params.mat',F_cc)

np.savetxt('F_cc_p18_4params_inv.mat',inv(F_cc))

np.savetxt('F_kc_p18_4params.mat',F_kc)

np.savetxt('F_kc_p18_4params_inv.mat',inv(F_kc))
'''




np.savetxt('F_ij_p18vHIHI_zH'+sys.argv[1]+'_bias1_7params.mat',F)

#np.savetxt('F_p18vHIHI_zH'+sys.argv[1]+'_bias1_inv.mat',inv(F))

#np.savetxt('F_kk_p18_bias1.mat',F_kk)

#np.savetxt('F_kk_p18_bias1_inv.mat',inv(F_kk))
'''
np.savetxt('F_HI6_9_p18v3_bias1.mat',F_HI6_9)

np.savetxt('F_HI6_9_p18v3_bias1_inv.mat',inv(F_HI6_9))

np.savetxt('F_HI9_9_p18v3_bias1.mat',F_HI9_9)

np.savetxt('F_HI9_9_p18v3_bias1_inv.mat',inv(F_HI9_9))

np.savetxt('F_HI13_9_p18v3_bias1.mat', F_HI13_9)

np.savetxt('F_HI13_9_p18v3_bias1_inv.mat', inv(F_HI13_9))
'''

'''
F_kk[0][0] = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_0_kk))[0][0]
F_cc[0][0] = np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_0_cc))[0][0]
F_kc[0][0] = np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_0_kc))[0][0]

F[0][0] = F_00


#=====================
F_01 = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_2_kk))[0][0]
F_01 += np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_2_cc))[0][0]
F_01 += np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_2_kc))[0][0]
F[0][1] = F_01
F[1][0] = F_01


F_kk[0][1] = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_2_kk))[0][0]
F_cc[0][1] = np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_2_cc))[0][0]
F_kc[0][1] = np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_2_kc))[0][0]

F_kk[1][0] = F_kk[0][1]
F_cc[1][0] = F_cc[0][1]
F_kc[1][0] = F_kc[0][1]

#======================
'''













































'''
F_02 = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_02 += np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_02 += np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]
F[0][2] = F_02
F[2][0] = F_02

F_kk[0][2] = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_cc[0][2] = np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_kc[0][2] = np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]

F_kk[2][0] = F_kk[0][2]
F_cc[2][0] = F_cc[0][2]
F_kc[2][0] = F_kc[0][2]

#=====================
F_03 = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_03 += np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_03 += np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]
F[0][3] = F_03
F[3][0] = F_03


F_kk[0][3] = np.matmul(u_0_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_cc[0][3] = np.matmul(u_0_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_kc[0][3] = np.matmul(u_0_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]

F_kk[3][0]=F_kk[0][3]
F_cc[3][0]=F_cc[0][3]
F_kc[3][0]=F_kc[0][3]


#=====================
F_11 = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_2_kk))[0][0]
F_11 += np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_2_cc))[0][0]
F_11 += np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_2_kc))[0][0]
F[1][1] = F_11


F_kk[1][1] = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_2_kk))[0][0]
F_cc[1][1] = np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_2_cc))[0][0]
F_kc[1][1] = np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_2_kc))[0][0]



#=====================
F_12 = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_12 += np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_12 += np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]
F[1][2] = F_12
F[2][1] = F_12


F_kk[1][2] = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_cc[1][2] = np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_kc[1][2] = np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]

F_kk[2][1] = F_kk[1][2]
F_kc[2][1] = F_kc[1][2]
F_cc[2][1] = F_cc[1][2]

#=====================
F_13 = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_13 += np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_13 += np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]
F[1][3] = F_13
F[3][1] = F_13

F_kk[1][3] = np.matmul(u_2_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_cc[1][3] = np.matmul(u_2_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_kc[1][3] = np.matmul(u_2_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]

F_kk[3][1]=F_kk[1][3]
F_cc[3][1]=F_cc[1][3]
F_kc[3][1]=F_kc[1][3]

#=====================
F_22 = np.matmul(u_3_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_22 += np.matmul(u_3_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_22 += np.matmul(u_3_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]
F[2][2] = F_22

F_kk[2][2] = np.matmul(u_3_kk.transpose(),np.matmul(C_kk_inv,u_3_kk))[0][0]
F_cc[2][2] = np.matmul(u_3_cc.transpose(),np.matmul(C_cc_inv,u_3_cc))[0][0]
F_kc[2][2] = np.matmul(u_3_kc.transpose(),np.matmul(C_kc_inv,u_3_kc))[0][0]


#=====================
F_23 = np.matmul(u_3_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_23 += np.matmul(u_3_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_23 += np.matmul(u_3_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]
F[2][3]=F_23
F[3][2]=F_23

F_kk[2][3] = np.matmul(u_3_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_cc[2][3] = np.matmul(u_3_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_kc[2][3] = np.matmul(u_3_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]

F_kk[3][2] = F_kk[2][3]
F_cc[3][2] = F_cc[2][3]
F_kc[3][2] = F_kc[2][3]
#=====================
F_33 = np.matmul(u_5_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_33 += np.matmul(u_5_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_33 += np.matmul(u_5_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]
F[3][3]=F_33

F_kk[3][3] = np.matmul(u_5_kk.transpose(),np.matmul(C_kk_inv,u_5_kk))[0][0]
F_cc[3][3] = np.matmul(u_5_cc.transpose(),np.matmul(C_cc_inv,u_5_cc))[0][0]
F_kc[3][3] = np.matmul(u_5_kc.transpose(),np.matmul(C_kc_inv,u_5_kc))[0][0]

#=====================



print F

print inv(F)

print np.matmul(F,inv(F))
np.savetxt('F_ij_planck18p_v2.mat',F)

np.savetxt('F_planck_18_inversep_v2.mat',inv(F))

np.savetxt('F_kk_v2.mat',F_kk)

np.savetxt('F_kk_inv_v2.mat',inv(F_kk))

np.savetxt('F_cc_v2.mat',F_cc)

np.savetxt('F_cc_inv_v2.mat',inv(F_cc))

np.savetxt('F_kc_v2.mat',F_kc)

np.savetxt('F_kc_inv_v2.mat',inv(F_kc))

'''