import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import cosmolopy.distance as cd
#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
#print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
#for name in powers: print(name)



#================================================ cross power
#For calculating large-scale structure and lensing results yourself, get a power spectrum
#interpolation object. In this example we calculate the CMB lensing potential power
#spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
#calling PK(z, k) will then get power spectrum at any k and redshift z in range.



for j in range(7): 
  cl_kappa=np.zeros((2,512*3))
  cl_HI = np.zeros((2,512*3))
  cl_cross = np.zeros((2,512*3))
  delta = np.zeros(7)
  sigma8 = np.zeros(2)
  print ("J =", j)
  for i in range (2):
    nz = 200 #number of steps to use for the radial/redshift integration
    kmax=10  #kmax to use
    #First set up parameters as usual
    delta_h = (-1**i)*0.0025
    #delta_h = (-1**i)*0.005
    pars = camb.CAMBparams()
    '''
    #============= WMAP 9
    omegac = np.zeros(6) + 0.233  #wmap 9
    omegab = np.zeros(6) + 0.046  #wmap 9
  
    h0 = np.zeros(6)+ 70.0/100.0  #wmap 9

    amp_s = np.zeros(6) + 0.06 #wmap
    opt_dept = np.zeros(6) + 0.06

    sp_index = np.zeros(6) + 0.97
    #===================
    '''
    omegac = np.zeros(7) + 0.233  
  
    omegab = 0.046
    h0 = np.zeros(7)+ 70.0/100.0  

    amp_s = np.zeros(7) + 2e-9 #wmap
   

    alpha = np.zeros(7) + 1.00

    sp_index = np.zeros(7) + 0.97

    b0 = np.zeros(7) + 0.67
    b1 = np.zeros(7) + 0.18
    b2 = np.zeros(7) + 0.05

    # h0,  Oc, ns, As, bias

    #===================== PLANCK 2018
    '''
    omegac = np.zeros(5) + 0.261  
    #omegab = np.zeros(6) + 0.0493  
    omegab = 0.0493
    h0 = np.zeros(5)+ 67.4/100.0  

    amp_s = np.zeros(5) + 2e-9 #wmap
    #opt_dept = np.zeros(6) + 0.06

    alpha = np.zeros(5) + 1.00

    sp_index = np.zeros(5) + 0.965
    '''
    #===================================

    h0[0] = h0[1] + h0[1]*0.01*(-1.)**i

    #omegab[1] = omegab[0] + omegab[0]*0.01*(-1.)**i

    omegac[1] = omegac[0] + omegac[0]*0.01*(-1.)**i

    amp_s[2] = amp_s[0] + amp_s[0]*0.05*(-1.)**i


    sp_index[3] = sp_index[0] + sp_index[0]*0.01*(-1.)**i
  
    #alpha[4] = alpha[0] + alpha[0]*0.01*(-1.)**i

    b0[4] = b0[0] + b0[0]*0.1*(-1.)**i

    b1[5] = b1[0] + b1[0]*0.1*(-1.)**i

    b2[6] = b2[0] + b2[0]*0.1*(-1.)**i

    print ("I=", i)
    print ("h0=", h0[j])
    print ("omegab=", omegab)
    print ("omegac=", omegac[j])
    print ("amp_s=", amp_s[j])
    print ("sp_index=", sp_index[j])
    print ("alpha=", alpha[j])
    delta[0] = h0[1]*0.01
    #delta[1] = omegab[0]*0.01
    delta[1] = omegac[0]*0.01
    delta[2] = amp_s[0]*0.05
    delta[3] = sp_index[0]*0.01
    #delta[4] = alpha[0]*0.01
    delta[4] = b0[0]*0.1
    delta[5] = b1[0]*0.1
    delta[6] = b2[0]*0.1



    #cos_pars = [h0*100, omegab*(h0**2),omegac*(h0**2),amp_s,opt_dept,sp_index] #H0, ombh2, omch2, mnu, tau, ns list

    pars.set_cosmology(H0=h0[j]*100, ombh2=omegab*(h0[j]**2), omch2 = omegac[j]*(h0[j]**2), mnu = 0.06, omk = 0, tau = 0.06)
    pars.InitPower.set_params(ns=sp_index[j],As=amp_s[j])

    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    Results = camb.get_results(pars)
    kh, z, pk = Results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    pars.NonLinear = model.NonLinear_both
    Results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = Results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    sigma8[i] = Results.get_sigma8()
    print ("sigma_8 = ", sigma8[i])

    
    #pars.set_cosmology(H0=h0*100, ombh2=omegab*(h0**2), omch2=omegac*(h0**2), mnu = 0.06+delta_h, omk = 0, tau = 0.06+delta_h)
    #pars.InitPower.set_params(ns=0.97+delta_h)

    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(pars)
    #chistar = results.conformal_time(0)- results.tau_maxvis
    #zstar = 0.7885 
    #zstar = 0.44
    zstar = 1.77
    chistar = results.comoving_radial_distance(zstar)
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)

    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    dz = (zs[2:]-zs[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
    #Here for lensing we want the power spectrum of the Weyl potential.
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
            hubble_units=False, k_hunit=False, kmax=kmax,
                var1='delta_tot',var2='delta_tot', zmax=zs[-1])

    #Have a look at interpolated power spectrum results for a range of redshifts
    #Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
    #plt.figure(figsize=(8,5))
    k=np.exp(np.log(10)*np.linspace(-4,2,200))
    zplot = [0, 0.5, 1, 4 ,20]
    '''
    for z in zplot:
     plt.loglog(k, PK.P(z,k))


    plt.xlim([1e-4,kmax])
    plt.xlabel('k Mpc')
    plt.ylabel('$P_\Psi\, Mpc^{-3}$')
    plt.legend(['z=%s'%z for z in zplot]);
    plt.show()
    '''

    #=================================== cl lensing

    #Get lensing window function (flat universe)
    z_HI =  np.loadtxt('zhislide')
   
    print('z_HI =', z_HI[int(sys.argv[1])])
    #print('z_HI[10] =', z_HI[10])   
   
    #for zi in range(16):
    H0 = h0[j]*100
    c = 3.0e5
    om = omegac[j]+omegab
    lensamp = 3*(om)*(H0**2)/(2.0*(c**2)) #lensing amplitude
    chib = results.comoving_radial_distance(zstar) # z(chib) = 0.44, 0.7885, 1.77 
    lensker = chis*(1+zs)*(chib-chis)/chib*lensamp # lensing kernel
      
    z0 = z_HI[int(sys.argv[1])]
    ohi = 0.00048+0.00039*zs-0.000065*(zs**2) # omega_HI

    #HIbias = alpha[j]*(0.67+0.18*zs+0.05*(zs**2)*dz/dchis)
    HIbias = b0[j] + b1[j]*zs + b2[j]*(zs**2)*dz/dchis
    cosmo = {'omega_M_0' : om, 'omega_lambda_0' : 1-om, 'h' : h0[j]}
    cosmo = cd.set_omega_k_0(cosmo)
    ez = cd.e_z(zs,**cosmo)
    T_bar = 180*ohi*h0[j]*(1+zs)**2/(ez) 
    #print T_bar
    HIker = T_bar*HIbias
    #win = (chistar-chis)**2/(chistar**2)  #lensing windows

    #(0.67+0.18*zs+0.05*(zs**2))
    #Do integral over chi
    ls = np.arange(0,512*3, dtype=np.float64)
    #cl_kappa=np.zeros((2,ls.shape))

    #cl_HI = np.zeros((2,ls.shape))
    #cl_cross = np.zeros((2,ls.shape))

    
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation

    chi_0 = results.comoving_radial_distance(z0)

    #print chi_0
    for i2, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        P = 1./(60.0*(2*np.pi)**0.5)*np.exp(-(chis-chi_0)**2/(2*(60.0**2)))
        #P = 1.0
        #P2 = 1./(150.0*(2*np.pi)**0.5)*np.exp(-(chis-chi_0)**2/(2*(150.0**2))) 
        
        #cl_HI[i][i2]  = np.dot(dchis, w*PK.P(zs, k, grid=False)*HIker*HIker*P*P/(chis*chis))
        #cl_kappa[i][i2] = np.dot(dchis, w*PK.P(zs, k, grid=False)*lensker*lensker/(chis*chis))
        cl_cross[i][i2] = np.dot(dchis, w*PK.P(zs, k, grid=False)*HIker*lensker*P/(chis*chis))
        
        #cl_HI2[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*HIker*HIker*P2*P2/(chis*chis))
        #cl_cross2[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*HIker*lensker*P2/(chis*chis))
  #cl_kappa_f = np.zeros(15)
  #cl_kappa_b = np.zeros(15)
  #cl_HI_f = np.zeros(15)
  #cl_HI_b = np.zeros(15)
  cl_cross_f = np.zeros(15)
  cl_cross_b = np.zeros(15)
  for ind in range(15):  
    #cl_kappa_f[ind] = cl_kappa[0][102*ind:102*(ind+1)].mean()
    #cl_kappa_b[ind] = cl_kappa[1][102*ind:102*(ind+1)].mean()

    #cl_HI_f[ind] = cl_HI[0][102*ind:102*(ind+1)].mean()
    #cl_HI_b[ind] = cl_HI[1][102*ind:102*(ind+1)].mean()

    #cl_cross_f[ind] = np.abs(cl_cross[0][102*ind:102*(ind+1)]).mean()
    #cl_cross_b[ind] = np.abs(cl_cross[1][102*ind:102*(ind+1)]).mean()

    cl_cross_f[ind] = cl_cross[0][102*ind:102*(ind+1)].mean()
    cl_cross_b[ind] = cl_cross[1][102*ind:102*(ind+1)].mean()
  if(j==2):
    delta[j]=(sigma8[0]-sigma8[1])*0.5
  #u_kappa = (cl_kappa_f-cl_kappa_b)/(2*delta[j])

  #u_HI = (cl_HI_f-cl_HI_b)/(2*delta[j])

  #u_cross = (np.abs(cl_cross_f)-np.abs(cl_cross_b))/(2*delta[j])
  u_cross = (cl_cross_f - cl_cross_b)/(2*delta[j])
  hp.write_cl('u_grad/wmap_'+str(j)+'_KHI_zi'+str(int(sys.argv[1])+1)+'_zs23.fits',u_cross)
  

