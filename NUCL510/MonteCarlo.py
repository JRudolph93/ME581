# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:39:26 2017
@author: JRRud
"""
# coding: utf-8

# In[1]:
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mplib
from numpy import log10
#import math

# In[2]:

def padList(target,template,pad = ''):    
    dL = len(template) - len(target)       
    if dL == 0:
        return target          
    elif dL > 0:
        if pad == 'matchLast':
            if len(target) > 0:
                pad = target[-1]
            else: 
                pad = ''
        return [*target, *[pad]*dL]        
    elif dL < 0:
        return target[0:len(template)]

def multiPlot(fns=[],fnLabels=[],fnStyles=[],\
              \
              points=[],pointlabels=[],pointstyles=[],\
              \
              xArrays=[],yArrays=[],arrayLabels=[],arrayStyles=[],lineWidth=2,\
              vlines=[],vlineLabels=[],vlineStyles=[],\
              \
              resolution=100, xaxis=True, size=[12,8],\
              title='',title_size=26,title_y=0.94,\
              \
              xaxislabel='x',yaxislabel='f(x)',\
              xmin=None, xmax=None, ymin=None, ymax=None,\
              show=True,xscale='linear',yscale='linear',**otherArgs):

    #plt.close()
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
    fig,ax = plt.subplots(1,1)
    if len(fns)==len(points)==0 and (len(yArrays)==0 or len(xArrays)==0): return None
    
    ############################### fns #################################    
    if len(fns) > 0:        

        if (xscale=='log' or xscale=='symlog'): 
            x_arr = 10**np.linspace(np.log10(xmin),np.log10(xmax),int(resolution + 1))
        else: x_arr = np.linspace(xmin,xmax,int(resolution + 1))
        np.seterr(divide='ignore',invalid='ignore')    
        fnLabels = padList(fnLabels,fns)
        fnStyles = padList(fnStyles,fns,pad='matchLast')
        for i,f in enumerate(fns):
            ax.plot(x_arr, f(x_arr), fnStyles[i], label=fnLabels[i])

    ############################# x and y arrays ##############################
    if len(xArrays) > 0 and len(yArrays) > 0:

        xArrays = padList(xArrays,yArrays,pad='matchLast')
        
        if ymax == None or ymin == None:
            if ymin == None: yminVal= max([np.min(array) for array in yArrays])
            else: yminVal = ymin
            if ymax == None: ymaxVal= max([np.max(array) for array in yArrays])
            else: ymaxVal = ymax
            
            delta = ymaxVal - yminVal
            ymin = yminVal - 0.1*delta
            ymax = ymaxVal + 0.1*delta
     
        if type(arrayLabels) != list: arrayLabels = [arrayLabels]
        else: arrayLabels = arrayLabels        
        if type(arrayStyles) != list: arrayStyles = [arrayStyles]
        else: arrayStyles = arrayStyles
        
        #x_arr = xArray
        np.seterr(divide='ignore',invalid='ignore')        
        for i,y_arr in enumerate(yArrays):            
            x_arr = xArrays[i]
            arrayLabels = padList(arrayLabels,yArrays)
            arrayStyles = padList(arrayStyles,yArrays,pad='matchLast')
            ax.plot(x_arr, y_arr, arrayStyles[i], label=arrayLabels[i],linewidth=lineWidth)

   
    ################################# points ##################################
    if points != []:  
        if type(points[0][0]) != list:
            pointSets = [points]
        else:
            pointSets = points

        for i,pointSet in enumerate(pointSets):
            xpt_list, ypt_list = pointSet[0],pointSet[1] 
            if len(pointSets)==len(pointlabels)==len(pointstyles):
                ax.plot(xpt_list,ypt_list,pointstyles[i],label=pointlabels[i])
            elif len(pointSets)==len(pointlabels): 
                ax.plot(xpt_list, ypt_list,'ko',label=pointlabels[i])
            elif len(pointSets)==len(pointstyles): 
                ax.plot(xpt_list, ypt_list, pointstyles[i]) 
            else: ax.plot(xpt_list,ypt_list,'ko')
            
    ################################## vlines #################################   
    for vline in vlines:
        xarr = np.array([vline,vline])
        if ymin != None and ymax != None: yarr = np.array([ymin,ymax])
        ax.plot(xarr, yarr, 'k--',linewidth=1) 
            
    ################################ add axes #################################      
            
    props = mplib.font_manager.FontProperties(size=22)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0: ax.legend(handles,labels,markerscale=1,prop=props)        
    if xaxis:
        x0_arr = np.array([xmin-1,xmax+1])
        y0_arr = np.array([0,0])
        ax.plot(x0_arr,y0_arr,'k--')
       
    ax.set_xlabel(xaxislabel,size=18,fontweight="bold")
    ax.set_ylabel(yaxislabel,size=18,fontweight="bold")  
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax],yscale=yscale)
    if type(xscale) == list and len(xscale) > 1:
        ax.set_xscale(value=xscale[0],**xscale[1])
    else:
        ax.set_xscale(value=xscale)
    ax.set(**otherArgs)
    fig.set_size_inches(size[0],size[1])
    ax.set_title(title, fontsize=title_size,fontweight="bold",y = title_y)

    if not show:
        plt.close()
        #plt.show(fig) 
        #print()
    #plt.close()
    return ax
    plt.show()
         #ax.set(xlim=[0.5, 4.5], ylim=[-2, 8],
# #      ylabel='Y-Axis', xlabel='X-Axis')

# In[4]:

def const(val):
    def const_(x):
        if type(x) == np.ndarray:
            return np.full_like(x,val)
        else:
            return val
    return const_

# In[8]:
    
def findRoot(f,xmin,xmax,zeroval=0,iterations=100,tolerance=1e-10):
    N,tol = iterations,tolerance
    a_i,b_i = xmin,xmax
    p_i = None
    for i in range(0,N):
        p_i = (a_i+b_i)/2.0   
        if i == N-1:            
            print("Last iteration; cycle complete.")
        else:
            fa_i = f(a_i) - zeroval
            fp_i = f(p_i) - zeroval
            if np.abs(fp_i) - tol <= 0.0:
                break
            elif ( fa_i * fp_i ) > 0:
                a_i = p_i
            elif ( fa_i * fp_i ) < 0: 
                b_i = p_i
            else:
                print("Something is wrong!","fa =",fa_i,"\nfp =",fp_i,"product =",fa_i * fp_i )
                break
    return p_i
    
def interpolate(xytuple1,xytuple2,xp):
    x1,y1 = xytuple1
    x2,y2 = xytuple2
    S = (y2-y1)/(x2-x1)
    yp = y1 + S*(xp-x1)
    return yp

def Newtons_P(xlist,ylist):        
    n = len(xlist)-1 
    Q = np.full([n+1,n+1],np.nan)    
    Q[:,0] = ylist    

    for i in range(1,n+1):
        for j in range(1,i+1):
            Q[i,j] = (Q[i,j-1]-Q[i-1,j-1])/(xlist[i]-xlist[i-j])
    a = [ Q[i,i] for i in range(0,n+1) ]        
        
    def P(x):            
        if type(x) == np.ndarray:
            y   = np.full_like(x,a[0])
        else: y = a[0]  
        
        for k in range(1,n+1):            
            term = a[k]            
            for i in range(0,k):
                term *= (x - xlist[i])
            y += term
        return y
    return P

# In[7]:

def integrate(fn,a,b,n=1000):   #definite integral    
    if a==b:
        return 0
    L = np.abs(b-a)
    dx = L/n
    mpList = [(fn(a+(i+1)*dx)+fn(a+i*dx))/2 for i in range(0,n)]
    Ival = 0
    for val in mpList:
        Ival += dx*val
    if b<a:
        Ival = Ival*(-1.0)
    return Ival

def interpolatedIntegral(fn,xmin=0,xmax=10,y0=0.0,Npoints=11,\
                         integralSteps=101,returnAll=False):  

    xvals = np.linspace(xmin,xmax,Npoints)
    Ivals = np.empty(Npoints,dtype=float)    
    for i in range(0,Npoints):
        Ivals[i] = integrate(fn,xvals[0],xvals[i],integralSteps) + y0
    if returnAll:  return Newtons_P(xvals,Ivals),xvals,Ivals
    else:          return Newtons_P(xvals,Ivals) 

def interpolatedInverse(fn,xmin=0,xmax=10,Npoints=9,\
                        domainCheckRes=100,returnAll=False):     
    deltax = xmax - xmin
    ffirst = fn(xmin)
    flast = fn(xmax) 
    deltaf = flast-ffirst
    incr = np.sign(deltaf)    
    dx = deltax/domainCheckRes
    fi = ffirst   
    
    goodIncrement = np.abs(incr)
    if goodIncrement:
        goodDomain = True
        for i in range(1,domainCheckRes+1):
            fnew = fn(xmin+i*dx)  
            if incr*(fnew - fi) <= 0:
                print("Domain error!")
                goodDomain = False
                break
            else: fi = fnew
    else: print("Incremenet error!")
        
    if goodIncrement and goodDomain:
        invarr = np.empty(Npoints)
        if incr > 0:            
            farr = np.linspace(ffirst,flast,Npoints)
            invarr[0] = xmin
            invarr[-1] = xmax
            
            for i in range(1,Npoints-1):
                invarr[i] = findRoot(fn,xmin=invarr[i-1],xmax=invarr[-1],\
                              zeroval=farr[i],iterations=20,tolerance=1e-5)
        elif incr < 0:            
            farr = np.linspace(flast,ffirst,Npoints)
            invarr[0] = xmax
            invarr[-1] = xmin
            
            for i in range(1,Npoints-1):
                invarr[i] = findRoot(fn,xmin=invarr[-1],xmax=invarr[i-1],\
                          zeroval=farr[i],iterations=20,tolerance=1e-5)
        else: print("Something is wrong.")
    
        inverse = Newtons_P(farr,invarr)     
        
    else:
        inverse = const(0)
        invarr = farr = np.empty(1)

    if returnAll: return inverse,farr,invarr
    else: return inverse 
    
# In[9]:

def rangeCheck(minVal,vals,maxVal,inclusive=[1,1]):
    incL,incR = inclusive
    gt = incL*(vals>=minVal) + (1-incL)*(vals>minVal)
    lt = incR*(vals<=maxVal) + (1-incR)*(vals<maxVal)
    return np.logical_and(gt,lt)

# In[10]:

Efmin,Efmax,ndivs = 0.1,6.0,1000

def chi_(E):
    chi_ = 0.453*np.e**(-1.036*E)*np.sinh(np.sqrt(2.29*E))
    return chi_

Ichi = integrate(chi_,Efmin,Efmax,ndivs)

def chi(E):
    #if np.logical_and(E>=Efmin, E<=Efmax):
    scaledChi = chi_(E) / Ichi
    if type(E)==np.ndarray:
        for i,val in enumerate(E):
            if not rangeCheck(Efmin,E[i],Efmax): scaledChi[i] = np.nan
    else:
        if not rangeCheck(Efmin,E,Efmax): scaledChi = np.nan
    return scaledChi


# In[27]:


multiPlot(fns=[chi_,chi,],fnLabels=['original chi(E)','scaled chi(E)'],fnStyles=[],\
          points=[], xmin=0, xmax=6.1, ymin=None, ymax=None,\
          xaxislabel='E (MeV)',yaxislabel='chi(E)', resolution=100,\
          title='Fission Neutron Energy Distribution',show=True)


Chi,xvs,yvs = interpolatedIntegral(chi,xmin=0.1,xmax=6,\
                                   Npoints=9,integralSteps=5000,returnAll=True)

multiPlot(fns=[Chi],fnLabels=[],\
          points=[xvs,yvs], xmin=0, xmax=6.1, ymin=None, ymax=None,\
          xaxislabel='E',yaxislabel='Chi(x)', resolution=100,\
          title='Integral of chi(E)',show=True)

inverseChi,ChiVals,EVals = interpolatedInverse(Chi,xmin=0.1,xmax=6,\
                                 Npoints=19,returnAll=True)

multiPlot(fns=[inverseChi],fnLabels=[],\
          points=[ChiVals,EVals], xmin=0, xmax=1, ymin=0, ymax=6,\
          xaxislabel='Chi',yaxislabel='E(Chi)', resolution=100,\
          title='Inverse Integral of Chi(E)',show=True)



def rmsChiTest(Emin,Emax,res):
    
    E = np.linspace(Emin,Emax,res+1)
    diff = (E - inverseChi(Chi(E)))**2
    diffMean = np.sum(diff)/(res+1)
    return np.sqrt(diffMean)



# In[26]:

def getGeneration(N=100):

    gen = np.empty(N)
    for i in range (0,N):
        gen[i] = inverseChi(np.random.rand())
    return gen

def tallyEnergies(arr,Emin,Emax,nBins,center=True,N=0):
    
    if N <= 0: N = len(arr)
    
    dE = (Emax - Emin) / nBins
    
    EArr = np.linspace(Emin,Emax-dE,nBins)
    NArr = np.zeros(nBins)

    for i in range(0,nBins):
        NArr[i] = sum( np.logical_and(arr >= EArr[i],arr < EArr[i]+dE) ) / (N*dE)
        #print(NArr[i], "counted b/w",EArr[i],"and",EArr[i]+dE)
    if center:
        EArr += np.full_like(EArr,dE/2)
    return np.array([EArr,NArr])  
  

# In[31]:

Eth = 10 #eV, top of thermal range
Es  = 1e5 #eV, top of slowing down range
Ef  = 6e6 #eV, top of fast range

sigmaS_H   = np.average([20, 4])*1e-24 #cm^2, representative microscopic scattering cross-section for H
sigmaS_O   = np.average([ 4, 3])*1e-24
sigmaS_U25 = np.average([10, 4])*1e-24
sigmaS_U28 = np.average([ 9, 5])*1e-24

# In[ ]:

Av = 6.0221409e23

class Component:
    
    def __init__(self,rho0=np.nan,M=np.nan,sigmaS=np.nan):
        self.rho0 = rho0
        self.M = M
        self.sigmaS = sigmaS
        self.N = np.nan
        
        Av = 6.0221409e23
        if rho0 != np.nan: self.N0 = self.rho0*Av/self.M
        else: self.N0 = np.nan
        
U235 = Component(M=235.0439299)
U238 = Component(M=238.05078826)
U    = Component(rho0=19.1)
H2O  = Component(rho0=1.0,M=18.01528)

def calculateDensities(enrichment=0.05,waterRatio=2):
    
    global U235,U238,U,H2O
    w = enrichment
    R = waterRatio    

    U.M  = w*U235.M + (1-w)*U238.M
    Av = 6.0221409e23
    U.N0 = U.rho0*Av/U.M

    alpha = H2O.N0*U.N0/(H2O.N0+R*U.N0)
    gamma = U238.M/U235.M * w/(1-w)
    beta1 = gamma/(gamma+1)
    beta2 = 1/(gamma+1)
    H2O.N  = alpha*R
    U235.N = alpha*beta1
    U238.N = alpha*beta2
    U.N  = U235.N + U238.N
    
calculateDensities()

# In[392]:



# In[393]:

resonances = np.array([1e4,2e4,3e4,4e4,5e4])

# In[420]:

def SigmaS(E=0):
    
    SigS = H2O.N*(2*sigmaS_H+sigmaS_O) + U235.N*sigmaS_U25 + U238.N*sigmaS_U28
    if type(E) == np.ndarray:
        return np.full_like(E,SigS)
    else:
        return SigS 

def SigmaA(Earr,returnRes=False):

    if type(Earr) != np.ndarray: 
        nonArray = True        
        Earr = np.array([Earr])
    else:
        nonArray = False
    Sarr = np.zeros_like(Earr,dtype=float)
    res = False
    for i,E in enumerate(Earr):
        
        if Eth <= E <= Es:
            resonant = False
            for E_res in resonances:
                h = 0.05*E_res
                if (E_res-h) <= E <= (E_res+h):
                    resonant = True
            if resonant:                
                Sarr[i] = 0.96/(1-0.96) * SigmaS()
            else:
                Sarr[i] = 0.01/(1-0.01) * SigmaS()
    
        elif 0 < E < Eth:        
            Sarr[i] = np.sqrt(Eth/E) * 0.01/(1-0.01) * SigmaS()
    
        elif E == 0:
            Sarr[i] = np.inf

    if nonArray:
        if returnRes:
            return Sarr[0],res
        else:
            return Sarr[0]
    else:
        if returnRes:
            return Sarr,res
        else:
            return Sarr
        
def SigmaA_logE(EArr):
    return np.array([SigmaA(10**Eval) for Eval in EArr])


def velocity(E):
    return 1.383e6*np.sqrt(E)

p0=0.1
def dtFunc(p0=0.1):
    
    def dt(E):
        v = velocity(E)
        SigmaT = SigmaS(E)+SigmaA(E)
        return -np.log(1-p0)/(SigmaT*v)
    return dt
dt1 = dtFunc(p0=0.1)
print("dt(1MeV) =",dt1(1e6))
print("dt(10eV) =",dt1(10))
# In[422]:

AbsPlot = multiPlot(fns=[SigmaA],fnLabels=[],fnStyles=['b-'],\
          points=[], xmin=0.001, xmax=1.5e5,ymin=-0.001, ymax=0.0001, xscale='symlog',yscale='linear',\
          xaxislabel='E (eV)',yaxislabel='SigmaA (cm^2)', resolution=500,vlines=[10,1e5,*resonances],\
          title='Macroscopic Absorption Cross Section',show=True)

AbsPlot = multiPlot(fns=[dtFunc(p0=0.1)],fnLabels=['dt'],fnStyles=['b-'],\
          points=[], xmin=0.01, xmax=6e6, ymin=1e-12, ymax=1e-6, xscale='log',yscale='log',\
          xaxislabel='E (eV)',yaxislabel='dt', resolution=500,vlines=[10,1e5,*resonances],\
          title='dt(E)',show=True)

# In[ ]:




class Generation:   

    vArr = np.nan
    logEArr = np.nan
    interactions = 0
    absorptions = 0
    scatters = 0
    
    def __init__(self,N=100):
        self.N = N
        self.E = np.empty(N)
        for i in range (0,N):
            self.E[i] = inverseChi(np.random.rand()) * 10**6
        #self.E = np.full(N,1e1)

    def step(self,dt):
        #print()
        for i in range(0,self.N):
            E = self.E[i]
            v = velocity(E)
            #print("Energy =",E)
            #print("Speed =",v)
            SigmaA_ = SigmaA(E)
            SigmaS_ = SigmaS()
            SigmaT = SigmaA_ + SigmaS_
            #tau = 1/(SigmaT*v)            
            P_interaction = 1 - np.e**(-SigmaT*v*dt) 
            
            r1 = rand()             
            if r1 <= P_interaction:
                self.interactions += 1
                
                P_abs = SigmaA_ / SigmaT                
                r2 = rand()                
                if r2 <= P_abs:
                    self.absorptions += 1
                    self.E[i] = 0

                else: 
                    self.scatters += 1                    
                    r3 = rand()
                    if E <= Eth:
                        newE = r3 * Eth
                    else:
                        newE = r3 * E                        
                    self.E[i] = newE        
            
        self.E = np.extract(self.E>0, self.E)
        self.N = len(self.E) 

        #print(self.absorptions, "out of",self.N,"neutrons were absorbed.")
        #print(self.scatters, "out of",self.N,"neutrons scattered.")
        #print("Number of neutrons remaining =",self.N)     
   
numPerGen = 5000    
gen1 = Generation(numPerGen)

gen1_orig = gen1.E[:]
print("Original N:",len(gen1_orig))

for i in range(0,0):
    gen1.step(1e-8)

gen1_mid = gen1.E[:]
print("Middle N:",len(gen1_mid))

for i in range(0,2000):
    gen1.step(1e-8)

gen1_last = gen1.E[:]
print("Final N:",len(gen1_last))


nbins1 = 45
Es1,tly1 = tallyEnergies(np.log10(gen1_orig),Emin=-2,Emax=7,nBins=nbins1,N=numPerGen)
tly2 = tallyEnergies(np.log10(gen1_mid),Emin=-2,Emax=7,nBins=nbins1,N=numPerGen)[1]
tly3 = tallyEnergies(np.log10(gen1_last),Emin=-2,Emax=7,nBins=nbins1,N=numPerGen)[1]

genPlot = multiPlot(fns=[],fnLabels=[],fnStyles=['k--'],\
          xArrays=[Es1,Es1,Es1],yArrays=[tly1,tly2,tly3],arrayStyles=['b.-','r.-','y.-'],\
          arrayLabels=['orig','middle','final'],lineWidth=1,\
          xaxislabel='log10(E)',yaxislabel=['N(E)'],vlines=[1,5,np.log10(1e5),np.log10(resonances)],\
          xaxis=True, title='Energy Distribution',\
          xmin=-2   , xmax=7, ymin=0, ymax=None,show=True)
genPlot.set_title('qwer')

######################################################################################################################
