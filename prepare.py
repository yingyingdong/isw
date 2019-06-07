import numpy as np
import sys
import healpy as hp
#xf = np.fft.fftshift(np.fft.fftn(x/xm-1.)
dra    = np.pi/2
boxize = 600.#1200.
simu   = '1'#'6514'#'6620'

def fft3d(ima):
    #return np.fft.fftshift(np.fft.fft2(ima))
    return np.fft.fftn(ima)

def fft3d_freq(nx,ny,nz):
    #freqz = np.roll(np.fft.fftfreq(nz),nz/2)#.reshape(ny,1)
    #freqy = np.roll(np.fft.fftfreq(ny),ny/2)#.reshape(ny,1)
    #freqx = np.roll(np.fft.fftfreq(nx),nx/2)
    stepx = boxize/nx
    freqx = np.fft.fftfreq(nx)*(2*np.pi/stepx)
    freqy = np.fft.fftfreq(ny)*(2*np.pi/stepx)
    freqz = np.fft.fftfreq(nz)*(2*np.pi/stepx)
    kx,ky,kz = np.meshgrid(freqx,freqy,freqz)
    k2 = kx*kx+ky*ky+kz*kz
    return k2
'''
def fft3d_freq(nx,ny,nz):
    #freqy = np.roll(np.fft.fftfreq(ny),ny/2)#.reshape(ny,1)
    #freqx = np.roll(np.fft.fftfreq(nx),nx/2)
    stepx,stepy,stepz = boxize/nx,boxize/ny,boxize/nz
    freqx = np.fft.fftfreq(nx)*(2*np.pi/stepx)
    kx,ky,kz = np.meshgrid(freqx,freqx,freqx)
    k2 = kx*kx+ky*ky+kz*kz
    return k2
'''
def fft2d(ima):
    #return np.fft.fftshift(np.fft.fft2(ima))
    return np.fft.fft2(ima)

def fft2d_freq(nx,ny):
    #freqy = np.roll(np.fft.fftfreq(ny),ny/2)#.reshape(ny,1)
    #freqx = np.roll(np.fft.fftfreq(nx),nx/2)
    stepx,stepy = boxize/nx,boxize/ny
    freqy = np.fft.fftfreq(ny)*(2*np.pi/stepy)#.reshape(ny,1)
    freqx = np.fft.fftfreq(nx)*(2*np.pi/stepx)
    kx = np.zeros(nx*ny).reshape(ny,nx)
    ky = np.zeros(nx*ny).reshape(ny,nx)
    for i in np.arange(ny):
        kx[i,:] = freqx
    for i in np.arange(nx):
        ky[:,i] = freqy
    k2 = kx*kx+ky*ky
    return kx,ky,k2

def operator(kx,ky,k2):
    return (ky*ky-kx*kx)/k2,2*kx*ky/k2

def merge_xyz(nx,ny,nz,nbin,h0):
    if(nbin==1):
        hxyz,nxx,nyy,nzz = h0.copy(),nx,ny,nz
    else:
        nxx,nyy,nzz = nx/nbin,ny/nbin,nz/nbin
        hxy = np.zeros((nxx,nyy,nzz))
        for i in np.arange(nxx):
            for j in np.arange(nyy):
                for k in np.arange(nzz):
                    hxyz[i,j] = np.mean(h0[i*nbin:(i+1)*nbin,j*nbin:(j+1)*nbin,k*nbin:(k+1)*nbin])
    return hxyz,nxx,nyy,nzz

def arr_3d_2d(h3,nx):
    h2 = np.zeros((nx,nx))
    for i in np.arange(nx):
        for j in np.arange(nx):
            h2[i,j] = np.sum(h3[i,j])
    return h2

def arr_3d_2d_xyz(h3,nx,flag_x):
    h2 = np.zeros((nx,nx))
    for i in np.arange(nx):
        for j in np.arange(nx):
            if(flag_x==0): h2[i,j] = np.sum(h3[:,i,j])
            if(flag_x==1): h2[i,j] = np.sum(h3[i,:,j])
            if(flag_x==2): h2[i,j] = np.sum(h3[i,j,:])
    return h2

def arr_3d_3d_xyz(h30,n1,n2,flag_x):
    if(flag_x==0):
        tmp = np.zeros((h30.shape[1],h30.shape[2],n2-n1))
        for i in np.arange(n2-n1):
            tmp[:,:,i] = h30[n1+i,:,:].copy()
    if(flag_x==1):
        tmp = np.zeros((h30.shape[0],h30.shape[2],n2-n1))
        for i in np.arange(n2-n1):
            tmp[:,:,i] = h30[:,n1+i,:].copy()
    if(flag_x==2):
        tmp = np.zeros((h30.shape[0],h30.shape[1],n2-n1))
        for i in np.arange(n2-n1):
            tmp[:,:,i] = h30[:,:,n1+i].copy()
    return tmp
 
def pos_xyz(pos,dl,d1,d2,flag_x):
    tmp = pos.copy()
    if(flag_x==0): #yz
        tmp[:,:2] = pos[:,1:].copy()
        tmp[:,2]  = pos[:,0] + dl
    if(flag_x==1): #xz
        tmp[:,2]  = pos[:,1] + dl
        tmp[:,1]  = pos[:,2].copy()
    if(flag_x==2): #xy
        tmp[:,2]  = tmp[:,2] + dl
    index = (tmp[:,2]>d1)*(tmp[:,2]<d2)
    return tmp[index]

def pos_xyz_rotate_slice(pos,n11,n21,d1,d2,boxize):
    if(n11==0 and n21==1):
        pos1 = pos_xyz(pos,0,d1,boxize,2)
        pos2 = pos_xyz(pos,boxize,boxize,d2,1)
        pos1 = np.append(pos1,pos2,0)
    if(n11==1 and n21==2):
        pos1 = pos_xyz(pos,boxize,d1,2*boxize,1)
        pos2 = pos_xyz(pos,2*boxize,2*boxize,d2,0)
        pos1 = np.append(pos1,pos2,0)
    if(n11==2 and n21==3):
        pos1 = pos_xyz(pos,2*boxize,d1,3*boxize,0)
    return pos1
def arr_delta(h):
    hm = np.mean(h)
    return h/hm-1

def heapy_map(theta,phi,nside,nest):
    pix = hp.ang2pix(nside,theta,phi,nest=nest)
    rmap = np.zeros(hp.nside2npix(nside)) #Blank healpix map
    np.add.at(rmap,pix,1)
    return rmap
