import numpy as np
import astropy
import healpy as hp
import astropy
from astropy.io import fits
import treecorr as tc
import prepare_hp as pre
au = 180./np.pi

flag_load,flag_save = 0,1
flag_group = 3
m1,m2 = -30, -22
#m1,m2 = -21.5, -21.2
z1,z2 = 0.01,0.6
#(1) load group catalogue
if(flag_group==3):
    if(flag_load==0):
        #ff = fits.open('../data/dr7_photoz_csp_totcat.fits')
        ##flag_g= np.load('../data/BASS_area.npy')
        #x  = ff[1].data
        #ira,idec,iz,izerr,imag = 'RA','DEC','photo_z','photo_zerr','MAG_R'
        x     = np.load('../data/BASS/dr7_photoz_csp_totcat_Magr.npy')
        ira,idec,iz,izerr,iMag,imag,imag_err = 0,1,2,3,4,5,6
        if(m1<0): index = (x[iMag]>m1)*(x[iMag]<m2)*(x[iz]>z1)*(x[iz]<z2)#*(flag_g==nn)
        if(m1>0): index = (x[imag]>m1)*(x[imag]<m2)*(x[iz]>z1)*(x[iz]<z2)#*(flag_g==nn)
        ra_t  = x[ira]
        dec_t = x[idec]
    nside_mask = 1024
    name_pre,name_file = 'bass','BASS'
    mask     = np.load('../data/'+name_pre+'mask'+np.str(nside_mask)+'.npy')
    fname_edge = 'bassldp-pos2galaxy20-21R2.npy'
    x_edge     = np.load('../data/'+name_file+'/'+fname_edge)

theta,phi = np.radians(90.-dec_t[index]),np.radians(ra_t[index])
nside_ldp = 4096
npix_ldp  = hp.nside2npix(nside_ldp)
id_ldp    = np.arange(npix_ldp)

Radius = 5#5#6 #arcmin
radius = np.radians(Radius/60.)
vec_g  = hp.ang2vec(theta,phi)

mask_ldp  = np.ones(npix_ldp)
if(nside_mask!=nside_ldp): 
    mask_new  = pre.mask_nside12(mask,nside_ldp,nside_mask)
for i in np.arange(theta.shape[0]):
    idx_search = hp.query_disc(nside_ldp,vec_g[i],radius,nest=True) #idx on grid map
    mask_ldp[idx_search] = 0
mask_ldp[mask_new==0] = 0
mask_ldp[np.int32(x_edge[2])] = 0
theta_ldp,phi_ldp = hp.pix2ang(nside_ldp,id_ldp[mask_ldp==1],nest=True)
ra,dec    = au*phi_ldp,90.-au*theta_ldp 
np.save('../data/'+name_file+'/'+name_pre+'ldp-pos'+np.str(Radius)+'galaxy'+np.str(m1)+np.str(m2)+'z'+np.str(z1)+'-'+np.str(z2)+'R'+np.str(Radius)+'.npy',np.vstack((ra,dec,id_ldp[mask_ldp==1])))
