import numpy as np
import healpy as hp

def heapy_map(theta,phi,nside,nest):
    pix = hp.ang2pix(nside,theta,phi,nest=nest)
    rmap = np.zeros(hp.nside2npix(nside)) #Blank healpix map
    np.add.at(rmap,pix,1)
    return rmap
def generate_mask(theta,phi,nside,nest):
    rmap = heapy_map(theta,phi,nside,nest)
    mask = np.zeros(hp.nside2npix(nside))
    mask[rmap>0] = 1
    return mask
def delta_map(theta1,phi1,nside1,theta2,phi2,nside2,nset):
    g_map = heapy_map(theta1,phi1,nside1,nest)
    mask  = generate_mask(theta2,phi2,nside2,nest)
    g_map = g_map[mask>0]
    g_mean= np.mean(g_map)
    g_map = g_map/g_mean-1
    return g_map

def divide_area_ori(theta,phi,nside):
    npix  = hp.nside2npix(nside)
    pix   = hp.ang2pix(nside,theta,phi,nest=True)
    flag_s = 0
    ncount = np.zeros(npix)
    flag   = np.zeros(pix.shape[0])
    for i in np.arange(npix):
        index = (pix==i)
        if(np.count_nonzero(index)>0):
            flag[index] = flag_s
            ncount[flag_s] = np.count_nonzero(index)
            flag_s = flag_s+1
    print flag_s
    return flag,ncount[:flag_s]

def divide_area_first(theta,phi,nside):
    npix  = hp.nside2npix(nside)
    pix   = hp.ang2pix(nside,theta,phi,nest=True)
    ncount = np.zeros(npix)
    flag   = np.zeros(pix.shape[0])
    for i in np.arange(npix):
        index = (pix==i)
        if(np.count_nonzero(index)>0):
            flag[index] = i
            ncount[i] = np.count_nonzero(index)
    return flag,ncount

def divide_area_second(theta,phi,nside):
    npix  = hp.nside2npix(nside)
    flag,ncount = divide_area_first(theta,phi,nside)
    nmean = np.mean(ncount[ncount>0])
    print(nmean)
    for i in np.arange(npix):
        if(ncount[i]>0 and ncount[i]<nmean/2):
            id_nei = hp.pixelfunc.get_all_neighbours(nside,theta=i,nest=True)
            #print(id_nei,ncount[id_nei])
            id_q   = np.where((ncount[id_nei]>0) & (ncount[id_nei]<nmean*1.5))
            if(np.count_nonzero(id_q)>0):
                print(id_nei,id_q)
                #print(id_nei[id_q[0][0]])
                id_qq = (ncount[id_nei[id_q]]==ncount[id_nei[id_q]].min())
                ncount[id_nei[id_q]][id_qq] = ncount[id_nei[id_q]][id_qq]+ncount[i]
                ncount[i] = 0
                flag[flag==i] = id_nei[id_q[0][0]]

    flag_s = 0
    flag_n = np.zeros(flag.shape[0])
    print(npix,flag_n.shape)
    ncount_n = np.zeros(npix)
    for i in np.arange(npix):
        index = (flag==i)
        if(np.count_nonzero(index)>0):
            flag_n[index] = flag_s
            ncount_n[flag_s] = np.count_nonzero(index)
            flag_s = flag_s+1
    print flag_s
    return flag_n,ncount_n[:flag_s]

def mask_nside12(mask,nside_g,nside_mask):
    mask_new  = np.ones(hp.nside2npix(nside_g))
    id_g      = np.arange(hp.nside2npix(nside_g))
    theta,phi = hp.pix2ang(nside_g,id_g,nest=True)
    id_gm     = hp.ang2pix(nside_mask,theta,phi,nest=True)
    index_m   = mask[id_gm]==0
    mask_new[index_m] = 0
    return mask_new

def thetaphi2radec(theta,phi):
    au = 180./np.pi
    ra,dec = phi*au,(np.pi/2-theta)*au
    index  = ra<0
    ra[index] = ra[index]+360.
    return ra,dec
def nside2thetaphi(nside,nest):
    npix_h= hp.nside2npix(nside)
    id_h  = np.arange(npix_h)
    theta_h,phi_h = hp.pix2ang(nside,id_h,nest=nest)
    return id_h,theta_h,phi_h
def nside2radec(nside,nest):
    id_h,theta_h,phi_h = nside2thetaphi(nside,nest)
    ra_h,dec_h  =  thetaphi2radec(theta_h,phi_h)
    return id_h,ra_h,dec_h,theta_h,phi_h
   

def radec2thetaohi(ra,dec):
    theta,phi = np.radians(90.-dec),np.radians(ra)
    return theta,phi

def id_nbins(nn,ng):
    step_g = ng/nn
    nbins  = np.zeros(nn+1)
    for i in np.arange(nn):
        nbins = step_g*i
    nbins[nn] = ng
    return nbins
