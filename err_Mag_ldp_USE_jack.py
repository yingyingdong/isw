import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
cmap = cm.jet

name_pre = 'BASS'
name_mag = '-30-22'
name_z   = 'z0.01-0.6'
radius   = '5'
rmax     = 'r10-600'#'r1-300'#'Rmax5'
jack     = 'jack4'
njack = 55#55#55#752#192
nr    = 15
nside = 'nside512nsidecmb512nside'+jack
name_use = 'random-'
xx    = np.zeros((njack,4,nr))
err   = np.zeros(nr)
xm = np.loadtxt('../data/'+name_pre+name_use+'-/bass'+name_mag+name_z+'R'+radius+rmax+'nbin'+np.str(nr)+nside+'.dat')
for i in np.arange(njack):
    xtmp = np.loadtxt('../data/'+name_pre+name_use+'/-bass'+name_mag+name_z+'ijack'+np.str(i)+'R'+radius+rmax+'nbin'+np.str(nr)+nside+'.dat')
    for j in np.arange(njack):
        if(i!=j):
            xx[j,0] = xx[j,0] + xtmp[0]*xtmp[2]
            xx[j,2] = xx[j,2] + xtmp[2]
for i in np.arange(njack):
    xx[i,0] = xx[i,0]/xx[i,2]
for i in np.arange(nr):
    err[i] = np.sqrt(njack)*np.sqrt(np.var(xx[:,0,i]))
    
'''
plt.title('ISW around LDP,'+name_z+',mag_r='+name_mag+',R_s'+radius+' arcmin')
plt.errorbar(xm[1],xm[0]*1e6,yerr=err*1e6,capsize=2,label='jack'+np.str(njack))
#xrand = np.loadtxt('../data/random/random19-20-1-300-5.dat')
#plt.plot(xrand[1],xrand[0]*1e6,label='700000 random points',color='grey',ls='--')
plt.axhline(0,ls='--',c='grey')
plt.xscale('log')
plt.legend()
plt.ylim(-1,0.5)
plt.xlabel('R[acrmin]')
plt.ylabel(r'Cross-corr[$\rm{\mu K}$]')
plt.tight_layout()
plt.savefig('ldp'+name_z+name_pre+name_mag+'R'+radius+nside+'.pdf')
'''
tmp = np.zeros((nr,nr))
xmm = np.zeros(nr)
for i in np.arange(nr):
    xmm[i] = np.mean(xx[:,0,i])
cov = np.zeros((nr,nr))
for i in np.arange(nr):
    for j in np.arange(nr):
        for k in np.arange(njack):
            tmp[i,j] = tmp[i,j]+(xx[k,0,i]-xmm[0])*(xx[k,0,j]-xmm[j])
xcov = (np.mat(tmp)).I
e1   = np.mat(xmm)
SNR = np.sqrt(e1*xcov*e1.T)
for i in np.arange(nr):
    for j in np.arange(nr):
        cov[i,j] = tmp[i,j]/np.sqrt(tmp[i,i])/np.sqrt(tmp[j,j])
vmin,vmax = 0.4,1.
norm = colors.Normalize(vmin=vmin, vmax=vmax)
X,Y=np.meshgrid(xm[1],xm[1])
Z  = cov
fs = 18
im = plt.pcolor(X,Y,Z, cmap=cmap,norm=norm)
plt.tick_params(labelsize=fs-2)
cbar = plt.colorbar(im)
cbar.set_ticks(np.linspace(vmin,vmax,10))
cbar.ax.tick_params(labelsize=fs-2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('R[arcmin]',fontsize=fs)
plt.ylabel('R[arcmin]',fontsize=fs)
plt.title('area='+np.str(njack)+',SNR='+np.str(SNR)[2:7],fontsize=fs)
plt.tight_layout()
