

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
F7 = np.loadtxt('F_ij_p18vk1.78_bias1.mat')
covariance1 = inv(F1)
covariance2 = inv(F2)
covariance3 = inv(F3)
covaraince4 = inv(F4)
coveraince5 = inv(F5)
coveraince6 = inv(F7)
covariance_total = inv(F1+F2+F3+F4+F5+F6+F7)
mean = [0.7, 0.3,  0.8, 0.97, 1.0] 
gauss1=GaussianND(mean, covariance1,names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss2=GaussianND(mean, inv(F1+F2),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss3=GaussianND(mean, inv(F1+F2+F3),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss4=GaussianND(mean, inv(F1+F2+F3+F4),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss5=GaussianND(mean, inv(F1+F2+F3+F4+F5),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss6=GaussianND(mean, inv(F1+F2+F3+F4+F5+F6),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss7=GaussianND(mean, inv(F7),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])

gauss8=GaussianND(mean, inv(F1+F2+F3+F4+F5+F6+F7),names=['h0','Om', 'sigma8','ns','alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}', r'\sigma_8',r'n_s',r'\alpha'])
print(covariance_total)
F_kk = np.loadtxt('F_kk_p18_4params.mat')

F_kc = np.loadtxt('F_kc_p18_bias1.mat')

F_cc = np.loadtxt('F_cc_p18_bias1.mat')

print (covariance_total)

#gauss_kk=GaussianND(mean, inv(F_kk),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])

gauss_cc=GaussianND(mean, inv(F_cc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

gauss_kc=GaussianND(mean, inv(F_kc),names=['h0','Om', 'sigma8','ns', 'alpha'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s', r'\alpha'])

#gauss_2=GaussianND(mean, inv(F_kc+F_cc),names=['h0','Om', 'sigma8','ns'], labels=[r'h_0', r'\Omega_\mathrm{m}',r'\sigma_8',r'n_s'])
settings = plots.GetDistPlotSettings()
settings.norm_prob_label = False
settings.axes_fontsize = 22
settings.lab_fontsize = 22
settings.figure_legend_loc = 'upper right'
settings.legend_fontsize = 18


g = plots.get_subplot_plotter(settings=settings)

#g.triangle_plot([gauss,gauss_kk,gauss_cc,gauss_kc,gauss_2],filled=True,legend_labels=['3x2pt','$\kappa\kappa$','HIHI','$\kappa$HI','2x2pt'])
#g.triangle_plot([gauss,gauss_cc,gauss_kc,gauss_kk],filled=True,legend_labels=['$3x2pt$','HIHI','$\kappa$HI','$\kappa\kappa$'])
g.triangle_plot([gauss6,gauss8],filled=True,legend_labels=['22x2pt','28x2pt(ks2)'])
#g.triangle_plot([gauss1,gauss2,gauss3,gauss4,gauss5,gauss6],filled=True,legend_labels=['3x2pt','6x2pt','10x2pt','14x2pt','18x2pt','22x2pt'])
#g.triangle_plot([gauss3],filled=True,legend_labels=['6x2pt'])
#plt.savefig('4params_planck_s8_3x2.png')
#plt.savefig('4params_planck_kk_HIHI.png')
#plt.savefig('4params_planck_sigma8.png')
#plt.savefig('4params_planck_kk_HIHI_kHI.png')
plt.savefig('bias1_planck_28x2k1.78_v2.png')

