import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
import cosmology as cosmos

def integrate_massfunc(xsort):
    nmass_func = 50
    I_mass_func = np.zeros(nmass_func)
    mass_min,mass_max = np.min(xsort),np.max(xsort)
    lg_step = (np.log10(mass_max)-np.log10(mass_min))/nmass_func
    lg_mass = lg_step*np.arange(nmass_func+1)+np.log10(mass_min)
    lg_mass_m = median_m(lg_mass)

    for i in np.arange(nmass_func):
        index = (np.log10(xsort)>=lg_mass[i])
        I_mass_func[i] = np.count_nonzero(index)
    plt.subplot(221)
    plt.plot(lg_mass_m,I_mass_func)
    plt.xlabel(r'$lg(Mh)-10$')
    plt.ylim(1e-1,1e7)
    plt.yscale('log')
    plt.legend(loc='best')
    #return I_mass_func

def gain_obs_number(x_obs,dv):
    xobs = x_obs[1]*dv
    index = xobs>0
    coefs2 = np.polyfit(x_obs[0,index], np.log10(xobs[index]), 10)
    ffit2 = np.polyval(coefs2, x_obs[0])
    plt.subplot(222)
    plt.plot(x_obs[0],xobs)
    plt.plot(x_obs[0], 10**ffit2,'x')
    plt.yscale('log')
    plt.legend(loc='best')
    return coefs2,np.min(x_obs[0,index])

class FINDM():
    def __init__(self,coefs2,xsort,Mag_min):
        self.mag_min,self.mag_max,self.n_mag = Mag_min,-19,60
        self.xmag = self.mag_min+np.arange(self.n_mag)*(self.mag_max-self.mag_min)/self.n_mag
        self.xm   = np.arange(12,15.5,0.01)
        self.ffit22 = np.polyval(coefs2,self.xmag)
        self.yxmag =  10**self.ffit22

        self.x_findm = np.zeros(self.n_mag)
        for i in np.arange(self.n_mag):
            if(self.yxmag[i]<xsort.shape[0]):
                self.x_findm[i] = np.log10(xsort[np.int32(np.floor(self.yxmag[i]))])+10
        self.index  = (self.yxmag<xsort.shape[0])*(self.xmag<-20)
        self.coefs3 = np.polyfit(self.xmag[self.index],self.x_findm[self.index], 20)
        self.coefs4 = np.polyfit(self.x_findm[self.index],self.xmag[self.index], 20)
        self.ffit3  = np.polyval(self.coefs3, self.xmag)
        self.ffit4  = np.polyval(self.coefs4, self.xm)
    def plt_mag(self):
        plt.subplot(223)
        plt.plot(self.xmag,self.x_findm,c='black')
        plt.plot(self.xmag,self.ffit3,'x',alpha=0.5,c='r')
        plt.plot(self.ffit4,self.xm,'x',alpha=0.5,c='g')
        plt.ylabel('Mh')
        plt.xlabel('MAG_i')
        plt.ylim(9,16)
        plt.xlim(-25,-18)
        plt.legend(loc='best')
        return self.coefs3,self.coefs4


#(1) simulation data
ol0,om0  = 0.732,0.268
simu,box = '1',600.
Nsnap='16'

dirh   = '/home/fydong/work/8100/cal_halo/data/subhalo_pos/'
fname2 = dirh+'current_mlimit-3_simu'+simu+'_Nsnap'+Nsnap+'.npy'
x = np.load(fname2)
median_m = lambda x:0.5*(x[1:]+x[:-1])
im = -1
    
#(1)halo data
xsort = (np.sort(x[:,im]))[::-1]
integrate_massfunc(xsort)
#check(xrange,dz,xsort,nhist)

#(2)load observe data
dz = 0.2
if(Nsnap=='21'): zcut = np.array([0.01,0.2])
if(Nsnap=='18'): zcut = np.array([0.2,0.4])
if(Nsnap=='16'): zcut = np.array([0.4,0.6])
zobs = np.str(("%g" % zcut[0]))+'-'+np.str(("%g" % zcut[1]))
x_obs = np.loadtxt('obs/R-InteMag-'+'z'+zobs+'.dat')
#plt.plot(x_obs[:,0],x_obs[:,1],label=zcut)

##(3)findm for halo mass
dv = box**3.
coefs2,Mag_min = gain_obs_number(x_obs,dv)
findm  = FINDM(coefs2,xsort,Mag_min)
coefs3,coefs4 = findm.plt_mag()
np.savetxt('abundance/simu'+simu+'-'+zobs+'Mh_Mag.dat',coefs3)
np.savetxt('abundance/simu'+simu+'-'+zobs+'Mag_Mh.dat',coefs4)



