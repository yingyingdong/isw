import numpy as np
import astropy
import healpy as hp
import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import cosmology as cosmos
import prepare_hp as pre
median_m = lambda x:0.5*(x[:-1]+x[1:])

class dC_z():
    def __init__(self,om0,ol0,ww):
        self.nz = 10000
        self.zmin  = 0.
        self.zmax  = 10.
        self.zstep = (self.zmax-self.zmin)/self.nz
        self.zbin  = self.zstep*np.arange(self.nz)+self.zmin
        self.dC = np.zeros(self.nz)
        for i in np.arange(self.nz):
            self.dC[i] = cosmos.d_comoving_w(self.zbin[i],ol0,om0,ww)
        self.dL = self.dC * (1+self.zbin)
    def Mag(self,zz,mag):
        index_z = np.int32(np.ceil((zz-self.zmin)/self.zstep))
        _dL     = self.dL[index_z]
        return mag-5*np.log10(_dL) - 25 #Magr-5lgh
    def Mag_complete(self,mag_limit,zz):
        _tmp    = self.Mag(zz,mag_limit)
        return _tmp
    def Mag_func(self,Mag,Mag_limit,vv):
        _nMag,_Mag_min = 50,Mag.min()
        _Mag_bin,_xmag = np.histogram(Mag,bins=_nMag,range=(-30,Mag_limit))
        _I_Mag_bin = np.zeros(_nMag)
        _xmag_m = median_m(_xmag)    
        for i in np.arange(_nMag):
            _I_Mag_bin[i] = np.sum(_Mag_bin[:i+1]/vv)
        return _xmag_m,_I_Mag_bin,_Mag_bin/(_xmag[1:]-_xmag[:-1])/vv
    
#This is for the r-band
om0,ol0,ww = 0.268,0.732,-1.
au    = 180./np.pi
nside = 512
flag_load  = 1
flag_group = 3
if(flag_group==3): name_pre = 'bass'
mask_new  = np.genfromtxt('../data/'+name_pre+'mask.dat')
mag_map   = np.load('../data/BASS/magr_limit-nside'+np.str(nside)+'.npy')

#(3) plot magr_limit distribution
#m2 = 23.
#tmp = mag_map.copy()
#tmp[mag_map<m2]=0
#hp.mollview(tmp,nest=True,title='mag_r>'+np.str(m2))

npix_h= hp.nside2npix(nside)
#(1) load group catalogue
if(flag_group==3):
    if(flag_load==0):
        ff   = fits.open('../data/dr7_photoz_csp_totcat.fits')
        #flag_g= np.load('../data/BASS_area.npy')
        x    = ff[1].data
        ira,idec,iz,izerr,imag,imag_err = 'RA','DEC','photo_z','photo_zerr','MAG_R','MAGERR_R'
        dcz  = dC_z(om0,ol0,ww)
        Magr = dcz.Mag(x[iz],x[imag])
        np.save('../data/BASS/dr7_photoz_csp_totcat_Magr.npy',np.vstack((x[ira],x[idec],x[iz],x[izerr],Magr,x[imag],x[imag_err])))
    if(flag_load==1):
        x    =  np.load('../data/BASS/dr7_photoz_csp_totcat_Magr.npy')
        ira,idec,iz,izerr,iMag,imag,imag_err = 0,1,2,3,4,5,6
        dcz  = dC_z(om0,ol0,ww)

fs = 15
#myc = ['indianred','orange','teal','steelblue']
myc = ['indianred','orange','teal','steelblue','coral']
#(2)calculate Magr function for different zbin
zbin = [0.01,0.2,0.4,0.6,0.8,1.2]
#for i in np.arange(zbin.shape[0]-1):
mask_ratio =  1.*np.count_nonzero(mask_new>0)/(mask_new.shape[0])
I_omega = 4*np.pi*mask_ratio
mag_limit = 23.

for i in np.arange(5)+0:
    label = 'bass,z:['+np.str(zbin[i])+','+np.str(zbin[i+1])+']'
    r1,r2 = cosmos.d_comoving_w(zbin[i],ol0,om0,ww),cosmos.d_comoving_w(zbin[i+1],ol0,om0,ww)
    vv = I_omega*1./3*(r2**3-r1**3) #(Mpc/h)^3
    Mag_limit = dcz.Mag_complete(mag_limit,zbin[i+1])
    index_g   = (x[iz]>zbin[i])*(x[iz]<zbin[i+1])
    
    # (1) histgram for zerr
    #zhist,zedge = np.histogram(x[izerr,index_g],bins=200,range=(0,0.6),density=True)
    #plt.bar(median_m(zedge),zhist,width=zedge[1:]-zedge[:-1],label=label,edgecolor=myc[i],color=myc[i],fill=True,linewidth=1.,alpha=0.6)
    #np.savetxt('../data/BASS/R-Mag-z'+np.str(zbin[i])+'-'+np.str(zbin[i+1])+'-zerr.dat',np.vstack((zedge[:-1],zedge[1:],zhist)))
    #index_g   = (x[iz]>zbin[i])*(x[iz]<zbin[i+1])*(x[izerr]<0.2)*(x[imag_err]<0.2)
    
    # (2) Mag func
    xmag_m,I_Mag_bin,Mag_bin = dcz.Mag_func(x[iMag,index_g],Mag_limit,vv)
    plt.plot(xmag_m,np.log10(Mag_bin),label=label)
    np.savetxt('../data/BASS/R-InteMag-z'+np.str(zbin[i])+'-'+np.str(zbin[i+1])+'.dat',np.vstack((xmag_m,I_Mag_bin)))
    np.savetxt('../data/BASS/R-Mag-z'+np.str(zbin[i])+'-'+np.str(zbin[i+1])+'.dat',np.vstack((xmag_m,Mag_bin)))
##plt.legend(fontsize=fs)
##plt.tick_params(labelsize=fs)
##plt.xlim(0,0.4)
##plt.xlabel(r'$\rm{z_{err}}$',fontsize=fs)
##plt.ylabel('PDF',fontsize=fs)
##plt.tight_layout()
##plt.savefig('z'+np.str(i)+'.pdf')

xsdss = np.loadtxt('SDSS_M.dat')
plt.errorbar(xsdss[:,0],xsdss[:,1],yerr=xsdss[:,2],label='SDSS,r band,z:[0.01,0.2]',ls='--',alpha=0.6)
plt.xlim(-24,-18)
plt.ylim(-5,-1)
plt.xlabel('Mag_R-5lgh')
plt.axvline(-19.5,label='-19.5',c='grey',ls='--',alpha=0.3)
plt.title('absolute Magnitude function')
plt.legend(fontsize=fs)
plt.tick_params(labelsize=fs)

#generate zerr random number according to the zerr pdf:
def yy_xhist(xh_min,step_xh,xx,yhist):
    index_xx = np.int32(np.floor((xx-xh_min)/step_xh))
    return yhist[index_xx]
def gene_ranf(x1,x2,nrand):
    return np.random.ranf(nrand)*(x2-x1)+x1
#Generate the random number according to a given PDF
def gene_random_pdf(xh_min,step_xh,yhist,nrand):
    xmin,xmax = 0,0.3 #xedge
    ymin,ymax = 0,np.max(yhist) #yedge
    xrand,yrand = gene_ranf(xmin,xmax,5*nrand), gene_ranf(ymin,ymax,5*nrand)
    index = yrand<yy_xhist(xh_min,step_xh,xrand,yhist)
    xrand,yrand = xrand[index],yrand[index]
    if(xrand.shape[0]<nrand):
        while(xrand.shape[0]<nrand):
            xtmp,ytmp = gene_ranf(xmin,xmax,nrand), gene_ranf(ymin,ymax,nrand)
            index     = ytmp<yy_xhist(xh_min,step_xh,xtmp,yhist)
            xrand,yrand = np.append(xrand,xtmp[index]),np.append(yrand,ytmp[index])
        return xrand[:nrand]
    else: return xrand[:nrand]

nrand = 10000
lw = 1
for i in np.arange(5):
    label = np.str(zbin[i])+'-'+np.str(zbin[i+1])
    xz = np.loadtxt('../data/BASS/R-Mag-z'+label+'-zerr.dat')
    xrand = gene_random_pdf(xz[0][0],(xz[1]-xz[0])[0],xz[2],nrand)
    plt.hist(xrand,normed=True,bins=30,histtype='step',color=myc[i],linewidth=2,alpha=0.6)
    plt.plot(0.5*(xz[0]+xz[1]),xz[2],label='z:'+label,ls='--',c=myc[i])

plt.legend()
plt.xlabel('zerr')
plt.ylabel('PDF')
plt.title('BASS')
