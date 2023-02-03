

# Show plots inline, and load main getdist plot module and samples class
from __future__ import print_function 
#%matplotlib inline config 
#%InlineBackend.figure_format = 'retina'
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
# use this *after* importing getdist if you want to use interactive plots
# %matplotlib notebook
import matplotlib.pyplot as plt
from numpy.linalg import inv
print('GetDist Version: %s, Matplotlib version: %s'%(getdist.__version__, plt.matplotlib.__version__))
# matplotlib 2 may not work very well without usetex on, can uncomment
#plt.rcParams['text.usetex']=True


from numpy.linalg import inv
from numpy.linalg import det
# Get some random samples for demonstration:
# make random covariance, then independent samples from Gaussian
import numpy as np
'''
ndim = 4
nsamp = 10000
np.random.seed(10)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)
A = np.random.rand(ndim,ndim)
cov = np.dot(A, A.T)
samps2 = np.random.multivariate_normal([0]*ndim, cov, size=nsamp)

# Get the getdist MCSamples objects for the samples, specifying same parameter
# names and labels; if not specified weights are assumed to all be unity
names = ["x%s"%i for i in range(ndim)]
labels =  ["x_%s"%i for i in range(ndim)]
samples = MCSamples(samples=samps,names = names, labels = labels)
samples2 = MCSamples(samples=samps2,names = names, labels = labels, label='Second set')



# Triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot([samples, samples2], filled=True)

plt.savefig('triangleplot.png')


from getdist import gaussian_mixtures
samples1, samples2 = gaussian_mixtures.randomTestMCSamples(ndim=2, nMCSamples=2)
g = plots.get_single_plotter(width_inch=4)
g.plot_1d([samples1, samples2], 'x0', marker=0)
plt.savefig('gaussian_mix.png')
'''


'''
from getdist.gaussian_mixtures import Mixture2D
from numpy.linalg import inv
cov1 = np.loadtxt('Q_h0Om.mat')
cov1 = inv(cov1)
print (cov1)
cov2 = [[0.001**2, -0.0006*0.05], [-0.0006*0.05, 0.05**2]]

#F = np.load('F_4params.mat.npy')

mean1 = [0.7, 0.3]
mean2 = [0.023, 0.09]





mixture=Mixture2D([mean1], [cov1], names=['h0','Om'], labels=[r'h_0', '\Omega_\mathrm{m}'], label='Model')

# Generate samples from the mixture as simple example
mix_samples = mixture.MCSamples(3000, label='Samples')

g = plots.get_subplot_plotter()
# compare the analytic mixture to the sample density
g.triangle_plot( mixture, filled=False)
plt.savefig('mixture2d.png')
'''




# The plotting scripts also let you plot Gaussian (or Gaussian mixture) contours 
from getdist.gaussian_mixtures import GaussianND

F = np.zeros((6,6))
F1 = np.zeros((6,6))
F2 = np.zeros((6,6))
F3 = np.zeros((6,6))
FP = np.zeros((6,6))
FP2 = np.zeros((6,6))
for i in range(16):
   F = F+np.loadtxt('F_ij_p18vHIHI_zH'+str(i+1)+'_bias1.mat')
   FP2 = FP2+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs23_bias1.mat')
   if(i<9):
     FP = FP+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs13_bias1.mat')
   if(i==3):
      F1=F1+F 
   if(i==7):
      F2=F2+F
   if(i==11):
      F3=F3+F


FT= F+FP+FP2
#FT= F+FP+FP2

FT2 = np.zeros((6,6))
FT2 = FT2+FT

for i in range(5):
    FT2 = FT2 + np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs8_bias1.mat')


''''
F1 = np.loadtxt('F_ij_p18_bias1.mat')
F2 = np.loadtxt('F_ij_p18v2_bias1.mat')
F3 = np.loadtxt('F_ij_p18v3_bias1.mat')
F4 = np.loadtxt('F_ij_p18vHI.13_bias1.mat')
F5 = np.loadtxt('F_ij_p18vHI.18_bias1.mat')
F6 = np.loadtxt('F_ij_p18vHI.23_bias1.mat')
F7 = np.loadtxt('F_ij_p18vk1.78_bias1.mat')
covariance1 = inv(F1)
covariance2 = inv(F2)
covariance3 = inv(F3)
covaraince4 = inv(F4)
coveraince5 = inv(F5)
coveraince6 = inv(F7)
'''
#covariance_total = inv(F1+F2+F3+F4+F5+F6+F7)
covariance = inv(F)
covarianceP = inv(FP)
covarianceP2 = inv(FP2)
covarianceT = inv(F+FP+FP2)
#print(covariance)
#print('\n\n ')
print((covarianceT[0][0]**0.5)*2,(covarianceT[1][1]**0.5)*2,(covarianceT[2][2]**0.5)*2,(covarianceT[3][3]**0.5)*2,(covarianceT[4][4]**0.5)*2,(covarianceT[5][5]**0.5)*2)
mean = [0.7, 0.3,  0.8, 0.97, 0.67, 0.19] 
'''
gauss1=GaussianND(mean, covariance1,names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss2=GaussianND(mean, inv(F1+F2),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss3=GaussianND(mean, inv(F1+F2+F3),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss4=GaussianND(mean, inv(F1+F2+F3+F4),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss5=GaussianND(mean, inv(F1+F2+F3+F4+F5),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss6=GaussianND(mean, inv(F1+F2+F3+F4+F5+F6),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss7=GaussianND(mean, inv(F7),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss8=GaussianND(mean, inv(F1+F2+F3+F4+F5+F6+F7),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])
'''
#print(covariance)
gauss1=GaussianND(mean, covariance,names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss2=GaussianND(mean, inv(F1),names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss3=GaussianND(mean, inv(F2),names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss4=GaussianND(mean, inv(F3),names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss5=GaussianND(mean, covarianceP,names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss6=GaussianND(mean, inv(F+FP),names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss7=GaussianND(mean, covarianceP2,names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss8=GaussianND(mean, covarianceT,names=['h0','Om', 'sigma8','ns','b0','b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
gauss9=GaussianND(mean, inv(FT2),names=['h0','Om', 'sigma8','ns','b0', 'b1'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'b_0',r'b_1'])
#F_kk = np.loadtxt('F_kk_p18_4params.mat')gauss1=GaussianND(mean, covariance,names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

#F_kc = np.loadtxt('F_kc_p18_bias1.mat')

#F_cc = np.loadtxt('F_cc_p18_bias1.mat')

#print (covariance)

#gauss_kk=GaussianND(mean, inv(F_kk),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])

#gauss_cc=GaussianND(mean, inv(F_cc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

#gauss_kc=GaussianND(mean, inv(F_kc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

#gauss_2=GaussianND(mean, inv(F_kc+F_cc),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])
settings = plots.GetDistPlotSettings()
settings.norm_prob_label = False
settings.axes_fontsize = 22
settings.lab_fontsize = 22
settings.figure_legend_loc = 'upper right'
settings.legend_fontsize = 18


g = plots.get_subplot_plotter(settings=settings)


#print(((inv(FT2))**0.5)*2.0)

#g.triangle_plot([gauss,gauss_kk,gauss_cc,gauss_kc,gauss_2],filled=True,legend_labels=['3x2pt','$\kappa\kappa$','HIHI','$\kappa$HI','2x2pt'])
#g.triangle_plot([gauss,gauss_cc,gauss_kc,gauss_kk],filled=True,legend_labels=['$3x2pt$','HIHI','$\kappa$HI','$\kappa\kappa$'])
#g.triangle_plot([gauss2,gauss3,gauss4,gauss1,gauss5,gauss6],filled=True,legend_labels=['4-HIHI','8-HIHI','12-HIHI','16-HIHI','9-KHI','16-HIHI+9-KHI'])
#g.triangle_plot([gauss1,gauss6,gauss8],filled=True,legend_labels=['16-HIHI','16-HIHI+9-KHI','16-HIHI+9Ks13HI+16Ks23HI'])
g.triangle_plot([gauss1,gauss6,gauss8,gauss9],filled=True,legend_labels=['16-HIHI','16-HIHI+1-$\kappa$','16-HIHI+2-$\kappa$','16-HIHI+3-$\kappa$'])
#g.triangle_plot([gauss1,gauss2,gauss3,gauss4,gauss5,gauss6],filled=True,legend_labels=['3x2pt','6x2pt','10x2pt','14x2pt','18x2pt','22x2pt'])
#g.triangle_plot([gauss3],filled=True,legend_labels=['6x2pt'])
#plt.savefig('4params_planck_s8_3x2.png')
#plt.savefig('4params_planck_kk_HIHI.png')
#plt.savefig('4params_planck_sigma8.png')
#plt.savefig('4params_planck_kk_HIHI_kHI.png')
plt.savefig('bias2_planck_16HIHI_3-K_v3.png')
plt.close()



#print(F-F3)
FP2=FT-FP2
F = F-F

FoM = np.zeros(18)
FoMPP = np.zeros(16)
FoMP = np.zeros(9)
for i in range(16):
   F = F+np.loadtxt('F_ij_p18vHIHI_zH'+str(i+1)+'_bias1.mat')
   FP2 = FP2+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs23_bias1.mat')
   FoM[i] = (F[1][1]*F[2][2]-F[1][2]*F[2][1])**0.5
   FoMPP[i] = (FP2[1][1]*FP2[2][2]-FP2[1][2]*FP2[2][1])**0.5 
   #FoM[i] = np.linalg.det(F)

FP = F+0
for i in range(9):
    FP = FP+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs13_bias1.mat')
    FoMP[i] = (FP[1][1]*FP[2][2]-FP[1][2]*FP[2][1])**0.5


FoM[16] = ((F[1][1]+FP[1][1])*(F[2][2]+FP[2][2]) - (F[1][2]+FP[1][2])*(F[2][1]+FP[2][1]))**0.5
FoM[17] = (FT[1][1]*FT[2][2]-FT[1][2]*FT[2][1])**0.5


x = np.arange(18)+1
xp2 = np.arange(16)+16+9
xp = np.arange(9)+16
x[16] = 16+9
x[17] = 16+9+16
y1 = FoM/(10**5)
y2 = FoMPP/(10**5)
y3 = FoMP/(10**5)
plt.plot(x[0:16],y1[0:16],'bo',label=r'$\sigma_8-\Omega_m$ 16HIHI')
#plt.plot(xp2,y2, 'g*', label=r'$\sigma_8-\Omega_m$ 16Ks23HI')
#plt.plot(xp,y3,'bo',label = '9Ks13HI')
plt.xlabel('# of redshift pairs')
plt.ylabel(r'FoM $\times$ $10^{-5}$')
#plt.ylabel('FoM')
plt.xticks()
plt.legend()
#plt.show()
plt.savefig('FoM_HIHI-bias2.png')
#plt.savefig('FoM_HIHI_9Ks13HI_bias2.png')


'''
FP2=FT2-FP2
F = F-F
FoM = np.zeros(16)
FoMPP = np.zeros(16)
FoMP = np.zeros(9)
FoMPPP = np.zeros(5)
for i in range(16):
   F = F+np.loadtxt('F_ij_p18vHIHI_zH'+str(16-i)+'_bias1.mat')
   FP2 = FP2+np.loadtxt('F_ij_p18vKHI_zH'+str(16-i)+'_zs23_bias1.mat')
   if i < 5:
      print(F)
   FoM[i] = (F[1][1]*F[2][2]-F[1][2]*F[2][1])**0.5
   FoMPP[i] = (FP2[1][1]*FP2[2][2]-FP2[1][2]*FP2[2][1])**0.5 


FP = F+0
for i in range(5):
    FP = FP+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs8_bias1.mat')
    FoMPPP[i] = (FP[1][1]*FP[2][2]-FP[1][2]*FP[2][1])**0.5 


for i in range(9):
    FP = FP+np.loadtxt('F_ij_p18vKHI_zH'+str(i+1)+'_zs13_bias1.mat')
    FoMP[i] = (FP[1][1]*FP[2][2]-FP[1][2]*FP[2][1])**0.5
#FoM[16] = ((F[1][1]+FP[1][1])*(F[2][2]+FP[2][2]) - (F[1][2]+FP[1][2])*(F[2][1]+FP[2][1]))**0.5
#FoM[17] = (FT[1][1]*FT[2][2]-FT[1][2]*FT[2][1])**0.5


x = np.arange(16)+1
xp = np.arange(5)+16+1
xp2 = np.arange(9)+16+5+1
xp3 = np.arange(16)+16+5+9+1
y1 = FoM/(10**5)
y2 = FoMPPP/(10**5)
y3 = FoMP/(10**5)
y4 = FoMPP/(10**5)
plt.plot(x,y1,'bo',label=r'$\sigma_8-\Omega_m$ 16HIHI')
plt.plot(xp,y2, 'g*', label=r'$\sigma_8-\Omega_m$ 5$\kappa$HI')
plt.plot(xp2,y3,'r+',label = r'$\sigma_8-\Omega_m$ 9$\kappa$HI')
plt.plot(xp3,y4,'kp', label = r'$\sigma_8-\Omega_m$ 16$\kappa$HI')
plt.xlabel('# of redshift pairs')
plt.ylabel(r'FoM $\times$ $10^{-5}$')
#plt.ylabel('FoM')
plt.xticks()
plt.legend(loc='lower right')
#plt.show()
plt.savefig('FoM_HIHI_9ks16ks5ks_bias2.png')
'''