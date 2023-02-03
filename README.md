# 2020s-KappaHI
This repository is used as the public available site for my paper submit to MNRAS.
The work focusing on the HI intensity map's forground removal effect on Lensing Convergence $\kappa$ and HI power spectra ($Kappa$HI)
We first generated the HI intensity map from the density-overdensity delta_m.
We use the N-body simulations to generate delta_m.
The N-body simulation and weak gravitational lensing catalogues can be obtained from http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky\_raytracing.
These catalogues also appear in Takahashi et al. 2017 https://ui.adsabs.harvard.edu/abs/2017ApJ...850...24T.

After create HI maps we use the use the foreground removal method presenting in our paper to remove the forground.
We then canculate the HIHI, $\kappa$$\kappa$ and $\kappa$HI powerspectra.
Afterward, we canculate the Fisher information matrice for our probs to constrain Cosmological parameters and HI bias.
