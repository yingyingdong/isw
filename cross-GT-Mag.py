import numpy as np
import astropy
import healpy as hp
import astropy
from astropy.io import fits
import treecorr as tc
import prepare_hp as pre

au = 180./np.pi
flag_load,flag_save = 0,1
flag_group  = 3
m1,m2       = -30, -21.5
z1,z2       = 0.01,0.2
#(1) load group catalogue
if(flag_group==3):
    if(flag_load==0):
        x     = np.load('../data/BASS/dr7_photoz_csp_totcat_Magr.npy')
        ira,idec,iz,izerr,iMag,imag,imag_err = 0,1,2,3,4,5,6
        index_g = (x[iMag]>m1)*(x[iMag]<m2)*(x[iz]>0.01)*(x[iz]<0.2)#*(flag_g==nn)
        ra_t  = x[ira]
        dec_t = x[idec]
    nside_mask = 256
    name_pre,name_file = 'bass','BASS'

if(flag_group==2):
    npoint = 10000000
    ra_t,dec_t = 360.*np.random.ranf(npoint),au*np.arcsin(1-np.random.ranf(npoint)*2)
    index_g    = (ra_t<361)
    nside_mask = 256
    name_pre,name_file = 'random','random'

if(flag_group==4):
    x = np.load('../data/SDSS7.npy')
    #flag_g= np.load('../data/SDSS7_area.npy')
    ira,idec,iz,imag = 2,3,4,5
    index_g = (x[:,imag]>m1)*(x[:,imag]<m2)#*(flag_g==nn)
    ra_t    = x[:,ira]
    dec_t   = x[:,idec]
    fname_m = '../data/rast_sdss_dr72safe0_nside512.fits'
    mask,sdss_h = hp.read_map(fname_m,nest=True,h=True)
    nside_mask = 512
    name_pre,name_file   = 'sdss','SDSS'


flag_plk  = 1
#(1)prepare galaxy map and generate mask for galaxies
nside = 256
if(flag_load == 0):
    if(flag_group==3):
        mask  = np.genfromtxt('../data/'+name_pre+'mask.dat')
        #mask  = pre.generate_mask(np.radians(90.-dec_t),np.radians(ra_t),nside_mask,True)
    theta,phi = np.radians(90.-dec_t[index_g]),np.radians(ra_t[index_g])
    g_map     = pre.heapy_map(theta,phi,nside,True)
    
    if(nside_mask != nside): mask_new  = pre.mask_nside12(mask,nside,nside_mask)
    else:       mask_new = mask
    index     = mask_new>0
    g_mean    = np.mean(g_map[index])
    g_map     = g_map/g_mean-1
    npix_g    = hp.nside2npix(nside)
    id_g      = np.arange(npix_g)
    theta,phi = hp.pix2ang(nside,id_g,nest=True)
    flag_area,ncount  = pre.divide_area_second(theta[index],phi[index])
    g_ra,g_dec,g_map  = au*phi[index],au*(np.pi/2-theta[index]),g_map[index]
    if(flag_save==1):
        np.savetxt('../data/'+name_pre+'-'+np.str(m1)+np.str(m2)+'z'+np.str(z1)+'-'+np.str(z2)+'.dat',np.vstack((g_ra,g_dec,g_map,flag_area)))
        #if(flag_group==3): np.savetxt('../data/'+name_pre+'mask.dat',mask_new)
if(flag_load == 1):
    g_ra,g_dec,g_map,flag_area = np.genfromtxt('../data/'+name_pre+'-'+np.str(m1)+'-'+np.str(m2)+'.dat') 
    #if(flag_group==3): mask_new = np.genfromtxt('../data/'+name_pre+'mask.dat')

#(2)prepare cmb map
if(flag_plk == 0):
    fname = '../data/wmap_band_iqusmap_r9_5yr_V_v3.fits'
    nside_cmb = 512
else:
    fname = '../data/COM_CMB_IQU-smica_2048_R3.00_hm1.fits'
    nside_cmb = 2048
cmb_map   = hp.read_map(fname,nest=True)
npix_cmb  = hp.nside2npix(nside_cmb)
id_cmb    = np.arange(npix_cmb)
r   = hp.rotator.Rotator(coord=['G','C']) # Transforms coordinates

cmb_theta,cmb_phi   = hp.pix2ang(nside_cmb,id_cmb,nest=True)
cmb_thetat,cmb_phit = r(cmb_theta,cmb_phi)
cmb_ra,cmb_dec      = (cmb_phit)*au,(np.pi/2-cmb_thetat)*au
index = cmb_ra<0
cmb_ra[index] = cmb_ra[index]+360
#index = cmb_map>0
#cmb_ra,cmb_dec = cmb_ra[index],cmb_dec[index]

#(3)calculate cross correlation
min_sep,max_sep = 1.,10*60.
nbins = 5

dataK = tc.Catalog(k=cmb_map, ra=cmb_ra, dec=cmb_dec, ra_units='deg', dec_units='deg')
datag = tc.Catalog(k=g_map,   ra=g_ra,   dec=g_dec,   ra_units='deg', dec_units='deg')
Kg = tc.KKCorrelation( nbins=nbins, min_sep=min_sep, max_sep=max_sep, bin_slop=0.01, verbose=0, sep_units='arcmin' )
Kg.process(dataK,datag,metric='Arc')
xim   = Kg.xi
rm    = Kg.meanr
min_name = '-'+np.str(np.int32(min_sep))+'-'+np.str(np.int32(max_sep))+'-'+np.str(np.int32(nbins))
np.savetxt('../data/'+name_file+'/'+name_pre+np.str(m1)+np.str(m2)+min_name+'z'+np.str(z1)+'-'+np.str(z2)+'.dat',np.vstack((Kg.xi,Kg.meanr,Kg.npairs)))

#(4)jackknife error bar
flag_max = np.int32(flag_area.max())
xi = np.zeros((flag_max,3,nbins))
for i in np.arange(flag_max):
    index = np.where(flag_area!=i)
    dataK = tc.Catalog(k=cmb_map, ra=cmb_ra, dec=cmb_dec, ra_units='deg', dec_units='deg')
    datag = tc.Catalog(k=g_map[index],ra=g_ra[index],dec=g_dec[index],ra_units='deg', dec_units='deg')
    Kg = tc.KKCorrelation( nbins=nbins, min_sep=min_sep, max_sep=max_sep, bin_slop=0.01, verbose=0, sep_units='arcmin' )
    Kg.process(dataK,datag,metric='Arc')
    xi[i,1] = Kg.xi
    xi[i,0] = Kg.meanr
    xi[i,2] = Kg.npairs
    np.savetxt('../data/'+name_file+'/'+name_pre+np.str(m1)+np.str(m2)+np.str(z1)+'-'+np.str(z2)+'ijack'+np.str(i)+min_name+'.dat',np.vstack((Kg.xi,Kg.meanr,Kg.npairs)))

#err = np.zeros(nbins)
#for i in np.arange(nbins):
#    err[i] = np.sqrt(np.var(xi[:flag_max,1,i]))*np.sqrt(flag_max)

