

# Show plots inline, and load main getdist plot module and samples class
from __future__ import print_function
from cProfile import label 
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
from numpy.linalg import det
print('GetDist Version: %s, Matplotlib version: %s'%(getdist.__version__, plt.matplotlib.__version__))
# matplotlib 2 may not work very well without usetex on, can uncomment
#plt.rcParams['text.usetex']=True



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

F1 = np.loadtxt('F_ij_p18_bias1.mat')
F2 = np.loadtxt('F_ij_p18v2_bias1.mat')
F3 = np.loadtxt('F_ij_p18v3_bias1.mat')
F4 = np.loadtxt('F_ij_p18vHI.13_bias1.mat')
F5 = np.loadtxt('F_ij_p18vHI.18_bias1.mat')
F6 = np.loadtxt('F_ij_p18vHI.23_bias1.mat')

### figure of merit for Om & Sigma8

# for F3, F5 and F6
F1_inv = inv(F1)
F2v2 = F1+F2
F2v2_inv = inv(F2v2) 
F3v2 =  F1+F2+F3
F3v2_inv = inv(F3v2)

F4v2 = F3v2+F4

F5v2 = F1+F2+F3+F4+F5
F5v2_inv = inv(F5v2)

F6v2 = F1+F2+F3+F4+F5+F6
F6v2_inv = inv(F6v2)


#1 2


FoM_Om_s8_1 =  (F1[1][1]*F1[2][2] - F1[1][2]*F1[2][1])**0.5
FoM_Om_s8_2 =  (F2v2[1][1]*F2v2[2][2] - F2v2[1][2]*F2v2[2][1])**0.5
FoM_Om_s8_3 = (F3v2[1][1]*F3v2[2][2] - F3v2[1][2]*F3v2[2][1])**0.5
FoM_Om_s8_4 = (F4v2[1][1]*F4v2[2][2] - F4v2[1][2]*F4v2[2][1])**0.5
FoM_Om_s8_5 = (F5v2[1][1]*F5v2[2][2] - F5v2[1][2]*F5v2[2][1])**0.5
FoM_Om_s8_6 = (F6v2[1][1]*F6v2[2][2] - F6v2[1][2]*F6v2[2][1])**0.5

FoM_Om_s8_1_v2 =  (F1_inv[1][1]*F1_inv[2][2] - F1_inv[1][2]*F1_inv[2][1])**-0.5
FoM_Om_s8_2_v2 =  (F2v2_inv[1][1]*F2v2_inv[2][2] - F2v2_inv[1][2]*F2v2_inv[2][1])**-0.5
FoM_Om_s8_3_v2 = (F3v2_inv[1][1]*F3v2_inv[2][2] - F3v2_inv[1][2]*F3v2_inv[2][1])**-0.5
FoM_Om_s8_5_v2 = (F5v2_inv[1][1]*F5v2_inv[2][2] - F5v2_inv[1][2]*F5v2_inv[2][1])**-0.5
FoM_Om_s8_6_v2 = (F6v2_inv[1][1]*F6v2_inv[2][2] - F6v2_inv[1][2]*F6v2_inv[2][1])**-0.5


# 2  4
FoM_s8_alpha_1 = (F1[1][1]*F1[4][4] - F1[1][4]*F1[4][1])**0.5
FoM_s8_alpha_3 = (F3v2[1][1]*F3v2[4][4] - F3v2[1][4]*F3v2[4][1])**0.5
FoM_s8_alpha_5 = (F5v2[1][1]*F5v2[4][4] - F5v2[1][4]*F5v2[4][1])**0.5
FoM_s8_alpha_6 = (F6v2[1][1]*F6v2[4][4] - F6v2[1][4]*F6v2[4][1])**0.5


#det all

FoM_1 = det(inv(F1))**-0.5
FoM_2 = det(inv(F1+F2))**-0.5
FoM_3 = det(inv(F1+F2+F3))**-0.5
FoM_5 = det(inv(F1+F2+F3+F4+F5))**-0.5
FoM_6 = det(inv(F1+F2+F3+F4+F5+F6))**-0.5



x = np.array([3,6,10,14,18,22])
y1 = np.array([FoM_Om_s8_1,FoM_Om_s8_2,FoM_Om_s8_3,FoM_Om_s8_4,FoM_Om_s8_5,FoM_Om_s8_6])
y1 = y1/(10**5)
y2 = np.array([FoM_Om_s8_1_v2,FoM_Om_s8_2_v2,FoM_Om_s8_3_v2,FoM_Om_s8_5_v2,FoM_Om_s8_6_v2])
y2 = y2/(10**5)

y3 = np.array([FoM_1,FoM_2,FoM_3,FoM_5,FoM_6])
plt.plot(x,y1,'bo',label=r'$\sigma_8-\Omega_m$')
#plt.plot(x,y2,'go',label=r'$\sigma_8-\Omega_m$ v2')
#plt.plot(x,y3,'bo')
plt.xlabel('# of redshift pairs')
plt.ylabel(r'FoM $\times$ $10^{-5}$')
#plt.ylabel('FoM')
plt.xticks()
plt.legend()
#plt.show()
plt.savefig('FoM.png')
#############################





covariance1 = inv(F1)
covariance2 = inv(F2)
covariance3 = inv(F3)
covaraince4 = inv(F4)
coveraince5 = inv(F5)
covariance_total = inv(F1+F2+F3+F4+F5+F6)
mean = [0.7, 0.3,  0.8, 0.97, 1.0] 
gauss1=GaussianND(mean, covariance1,names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss2=GaussianND(mean, inv(F1+F2),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss3=GaussianND(mean, inv(F1+F2+F3),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss4=GaussianND(mean, inv(F1+F2+F3+F4),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss5=GaussianND(mean, inv(F1+F2+F3+F4+F5),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss6=GaussianND(mean, covariance_total,names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

#print(covariance_total)
F_kk = np.loadtxt('F_kk_p18_4params.mat')

F_kc = np.loadtxt('F_kc_p18_bias1.mat')

F_cc = np.loadtxt('F_cc_p18_bias1.mat')

#print (covariance_total)

#gauss_kk=GaussianND(mean, inv(F_kk),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])

#gauss_cc=GaussianND(mean, inv(F_cc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

#gauss_kc=GaussianND(mean, inv(F_kc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

#gauss_2=GaussianND(mean, inv(F_kc+F_cc),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])
settings = plots.GetDistPlotSettings()
settings.norm_prob_label = False
settings.axes_fontsize = 26
settings.lab_fontsize = 26
settings.figure_legend_loc = 'upper right'
settings.legend_fontsize = 26


g = plots.get_subplot_plotter(settings=settings)

#g.triangle_plot([gauss,gauss_kk,gauss_cc,gauss_kc,gauss_2],filled=True,legend_labels=['3x2pt','$\kappa\kappa$','HIHI','$\kappa$HI','2x2pt'])
#g.triangle_plot([gauss,gauss_cc,gauss_kc,gauss_kk],filled=True,legend_labels=['$3x2pt$','HIHI','$\kappa$HI','$\kappa\kappa$'])
#g.triangle_plot([gauss3,gauss4,gauss5,gauss6],filled=True,legend_labels=['10x2pt','14x2pt','18x2pt','22x2pt'])
#g.triangle_plot([gauss1,gauss2,gauss3,gauss4,gauss5,gauss6],filled=True,legend_labels=['3x2pt','6x2pt','10x2pt','14x2pt','18x2pt','22x2pt'])
#g.triangle_plot([gauss4,gauss5,gauss6],filled=True,legend_labels=['14x2pt','18x2pt','22x2pt'])
#plt.show()
#g.triangle_plot([gauss3],filled=True,legend_labels=['6x2pt'])
#plt.savefig('4params_planck_s8_3x2.png')
#plt.savefig('4params_planck_kk_HIHI.png')
#plt.savefig('4params_planck_sigma8.png')
#plt.savefig('4params_planck_kk_HIHI_kHI.png')
#plt.savefig('bias1_planck_22x2_v2.png')

