import numpy as np
import astropy
import healpy as hp
import astropy
from astropy.io import fits
import treecorr as tc
import prepare_hp as pre

Radius     = '5'
flag_load  = 1
flag_group = 3
m1,m2      = '-30','-22'#np.sys.argv[1], np.sys.argv[2]
z1,z2      = '0.01','0.6'
M1,M2,Z1,Z2= np.float(m1),np.float(m2),np.float(z1),np.float(z2)
flag_plk   = 0
au         = 180./np.pi
nside      = 512
nside_mask = 512
nside_cmb  = 512
nside_jack = 4

#load mask files:
name_pre = 'bass'
mask = np.load('../data/'+name_pre+'mask'+np.str(nside_mask)+'.npy')
if(nside_mask != nside): mask_new = pre.mask_nside12(mask,nside,nside_mask)
else:                    mask_new = mask

if(flag_group==3): #ldp
    name_load,name_save = '../data/BASS/'+name_pre+'ldp-pos'+Radius+'galaxy'+m1+m2+'z'+z1+'-'+z2+'R'+Radius,'ldp-'
if(flag_group==1): #galaxy
    name_load,name_save = '../data/BASS/dr7_photoz_csp_totcat_Magr','gal-'
if(flag_group==2): #random
    name_load,name_save = '../data/BASS/'+name_pre+'-random','random-'

class GENE_MAP():
    def __init__(self,nside,mask):
        self.nside = nside
        self.mask  = mask
    def gene_map(self,ra,dec):
        theta,phi  = np.radians(90.-dec),np.radians(ra)
        g_map      = pre.heapy_map(theta,phi,self.nside,True)
        index      = self.mask>0
        g_mean     = np.mean(g_map[index])
        g_map      = g_map/g_mean-1
        npix_g     = hp.nside2npix(self.nside)
        id_g       = np.arange(npix_g)
        theta,phi  = hp.pix2ang(self.nside,id_g[index],nest=True)
        g_map      = g_map[index]
        return theta,phi,g_map

gene = GENE_MAP(nside,mask_new)
#(1) load catalogue
#(a) load LDPs for BASS
if(flag_group==3):
    if(flag_load==0):
        ra,dec,idx = np.load(name_load+'.npy')
        theta,phi,g_map  = gene.gene_map(ra,dec)
        #check for every ldp postion about the arrounding mask
        if(flag_group==3):
            flag_use   = np.ones(theta.shape[0])
            for i in np.arange(theta.shape[0]):
                id_nei = hp.pixelfunc.get_all_neighbours(nside,theta=theta[i],phi=phi[i],nest=True)
                if(np.count_nonzero(mask_new[id_nei]==0)>0): flag_use[i] = 0
            index = flag_use>0
            theta,phi,g_map = theta[index],phi[index],g_map[index]
            np.save(name_load+np.str(nside)+'.npy',np.vstack((theta,phi,g_map)))
    if(flag_load==1): 
        theta,phi,g_map = np.load(name_load+np.str(nside)+'.npy')

#(b) load BASS galaxies
if(flag_group==1):
    x     = np.load(name_load+'.npy')
    ira,idec,iz,izerr,iMag,imag,imag_err = 0,1,2,3,4,5,6
    index_g = (x[iMag]>M1)*(x[iMag]<M2)*(x[iz]>Z1)*(x[iz]<Z2)
    theta,phi,g_map = gene.gene_map(x[ira,index_g],x[idec,index_g]) 

#(c) gene BASS random points
if(flag_group==2):
    ra,dec = np.load(name_load+'.npy')
    theta,phi,g_map = gene.gene_map(ra,dec)
    #(2) divide into smaller areas
flag_area,ncount  = pre.divide_area_second(theta,phi,nside_jack)
g_ra,g_dec,g_map  = au*phi,au*(np.pi/2-theta),g_map

#(3)prepare cmb map
if(flag_plk == 0):
    cmb_map  = hp.read_map('../data/wmap_band_iqumap_r9_9yr_V_v5.fits',nest=True)
    cmb_mask = hp.read_map('../data/wmap_temperature_kq75_analysis_mask_r9_9yr_v5.fits',nest=True)
    nside_cmb = 512
else:
    if(nside_cmb == 2048):
        cmb_map = hp.read_map( '../data/COM_CMB_IQU-smica_2048_R3.00_hm1.fits',nest=True)
    if(nside_cmb < 2048): 
        cmb_map = np.load('../data/COM_CMB_IQU-smica_'+np.str(nside_cmb)+'_R3.00_hm1.npy')
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
#considering mask
index_mask = np.where(cmb_mask>0)
cmb_map,cmb_ra,cmb_dec = cmb_map[index_mask]/1e3,cmb_ra[index_mask],cmb_dec[index_mask]
cmb_map = (cmb_map-np.mean(cmb_map))
#(4)calculate cross correlation
rmin,rmax  =  10,15*60
min_sep,max_sep = rmin,rmax
nbins = 20

dataK = tc.Catalog(k=cmb_map, ra=cmb_ra, dec=cmb_dec, ra_units='deg', dec_units='deg')
datag = tc.Catalog(k=g_map,   ra=g_ra,   dec=g_dec,   ra_units='deg', dec_units='deg')
Kg = tc.KKCorrelation( nbins=nbins, min_sep=min_sep, max_sep=max_sep, bin_slop=0.01, verbose=0, sep_units='arcmin' )
Kg.process(dataK,datag,metric='Arc',num_threads=32)
xim   = Kg.xi
rm    = Kg.meanr
np.savetxt('../data/BASS/'+name_save+name_pre+m1+m2+'z'+z1+'-'+z2+'R'+Radius+'r'+np.str(rmin)+'-'+np.str(rmax)+'nbin'+np.str(nbins)+'nside'+np.str(nside)+'nsidecmb'+np.str(nside_cmb)+'nsidejack'+np.str(nside_jack)+'wmap.dat',np.vstack((Kg.xi,Kg.meanr,Kg.npairs,np.sqrt(Kg.varxi))))

#(5)jackknife error bar
flag_max = np.int32(flag_area.max())
xi = np.zeros((flag_max,2,nbins))
for i in np.arange(flag_max):
    #index = np.where(flag_area!=i)
    index = np.where(flag_area==i)
    dataK = tc.Catalog(k=cmb_map, ra=cmb_ra, dec=cmb_dec, ra_units='deg', dec_units='deg')
    datag = tc.Catalog(k=g_map[index],ra=g_ra[index],dec=g_dec[index],ra_units='deg', dec_units='deg')
    Kg = tc.KKCorrelation( nbins=nbins, min_sep=min_sep, max_sep=max_sep, bin_slop=0.01, verbose=0, sep_units='arcmin' )
    Kg.process(dataK,datag,metric='Arc',num_threads=32)
    xi[i,1] = Kg.xi
    xi[i,0] = Kg.meanr
    np.savetxt('../data/BASS/'+name_save+name_pre+m1+m2+'z'+z1+'-'+z2+'ijack'+np.str(i)+'R'+Radius+'r'+np.str(rmin)+'-'+np.str(rmax)+'nbin'+np.str(nbins)+'nside'+np.str(nside)+'nsidecmb'+np.str(nside_cmb)+'nsidejack'+np.str(nside_jack)+'wmap.dat',np.vstack((Kg.xi,Kg.meanr,Kg.npairs,np.sqrt(Kg.varxi))))

#(1)cmb_new = hp.ud_grade(cmb_map,512,order_in='nest',order_out='nest')
#(2)step = (np.log10(rmax)-np.log10(rmin))/5
#10.**(np.log10(rmin)+step*np.arange(nbins+1))
#(3)generate_mask(theta,phi,nside,nest)
