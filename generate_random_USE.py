import numpy as np
import healpy as hp
import prepare_hp as pre

au = 180./np.pi
npoint = 20000000 
nside  = 512
name_pre,name_file = 'bass','BASS'
ra,dec    = 360.*np.random.ranf(npoint),au*np.arcsin(1-np.random.ranf(npoint)*2)
theta,phi = np.radians(90.-dec),np.radians(ra)
mask = np.load('../data/'+name_pre+'mask'+np.str(nside)+'.npy')

id_nside = hp.ang2pix(nside,theta,phi,nest=True)
index    = mask[id_nside]==1
np.save('../data/'+name_file+'/'+name_pre+'-random.npy',np.vstack((ra[index],dec[index])))
