import numpy as np
import astropy
import healpy as hp
import astropy
from astropy.io import fits
import treecorr as tc
import prepare_hp as pre
au = 180./np.pi

nside_mask = 8192
nside_map  = 512

class mask_old_new():
    def __init__(self,nsideo,nsiden):
        self.nsideo  =  nsideo
        self.nsiden  =  nsiden
        self.dir     =  '/home/dfy/data/DelCals/'
        self.mask    =  np.load(self.dir+'mask_combine_nside8192.npy')
    def query_mask(self,nest):
        self.npixo = hp.nside2npix(self.nsideo)
        self.pix   = np.arange(self.npixo)
        self.index_mask = np.where(self.mask==0)
        self.thetao,self.phio = hp.pix2ang(self.nsideo,self.pix[self.index_mask],nest=nest)
        self.npixn = hp.nside2npix(self.nsiden)
        self.nmask = pre.heapy_map(self.thetao,self.phio,self.nsiden,nest)

Mask = mask_old_new(nside_mask,nside_map)
Mask.query_mask(True)
hp.mollview(Mask.nmask,nest=True)
np.save(Mask.dir+'Mask_nmask.npy',Mask.nmask)
plt.hist(Mask.nmask[Mask.nmask<256],bins=30)

nmask = 50
index_Mask = np.where((Mask.nmask<nmask))
map_temp   = np.zeros(Mask.npixn)
map_temp[index_Mask] = Mask.nmask[index_Mask].copy()
hp.mollview(map_temp,nest=True)
plt.savefig('fig/Mask_nmask'+np.str(nmask)+'.eps')

mask_temp = np.zeros(Mask.npixn)
mask_temp[map_temp>0] = 1 
hp.mollview(mask_temp,nest=True)
np.save(Mask.dir+'Mask_cut'+np.str(nmask)+'.npy',mask_temp)
plt.savefig('fig/Mask_cut'+np.str(nmask)+'.eps')
#hp.gnomview(Mask.mask,rot=[0,0],nest=True,xsize=1000,reso=3)
