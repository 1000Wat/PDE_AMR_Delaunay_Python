import numpy as np
import scipy.sparse as sp
import itertools as it
import numpy.linalg as lin
import scipy.interpolate as int
import matplotlib.pyplot as plt
import scipy.spatial as spa
import inspect as ins
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import time as ti
pct=0.1
NamesDiff={"dt":0,"dx":1,"dy":2,"dz":3}
DiffNames={0:"dt",1:"dx",2:"dy",3:"dz"}




class Grid:
    def GeneGridUniEuclid(self,NbPoint,dX0):
        self.dX0=dX0
        self.NbPoint=NbPoint
        self.co=np.zeros((np.product(NbPoint),np.size(dX0)))
        bound=[]
        signal=[]
        ranges=[range(NbPoint[i]) for i in range(NbPoint.size)]
        for i,xs in enumerate(it.product(*ranges)):
            # print(xs)
            coor=np.array(xs)*dX0
            self.co[i,:]=coor
            xs_t=xs[0]
            xs_x=xs[1]
            if xs_t%NbPoint[0]<=1  or xs_x%NbPoint[1]<=1 or xs_x%NbPoint[1]>=NbPoint[1]-2:# or xs_t%NbPoint[0]>=NbPoint[0]-2:
                bound.append(i)
            else:
                signal.append(i)
        bound=np.array(bound)
        self.bound=bound
        signal=np.array(signal)
        self.signal=signal
        

    def __init__(self,F,time_dir=False) -> None:
        self.F=F
        self.argF=ins.signature(F).parameters.keys()
        print(self.argF)
        self.time_dir=time_dir

    def BoundPurge(self,Array,Coord):
        return [Array[i] for i in Coord if i not in self.bound]

    def SetValueInit(self,f,arg):
        self.argdict={}
        TempPhi=np.zeros(self.co.shape[0])
        for i,co in enumerate(self.co):
            TempPhi[i]=f(co)
        self.argdict[arg]=TempPhi

    def TrigGrid(self):
        self.trig=spa.Delaunay(self.co,incremental=True)
            
    def PlotPhi2DScalar(self,ax:plt.Axes,arg,cmap="hot"):
        X=self.co[:,0]
        Y=self.co[:,1]
        Z=self.argdict[arg]
        ax.plot_trisurf(X,Y,Z,cmap =cmap)
    

        
    def Interp(self,arg,x:np.ndarray,argdict=None):
        if argdict is None:
            argdict=self.argdict
        simp=self.trig.find_simplex(x)
        Vertex=self.trig.simplices[simp]
        # print(simp)
        # print(Vertex)
        ArraySolve=np.ones((x.shape[0],x.shape[-1]+1))
        ArraySolve[...,1:]=x
        # print(ArraySolve)
        MaxtrixSolve=np.ones((ArraySolve.shape[0],ArraySolve.shape[-1],ArraySolve.shape[-1]))
        A=self.co[Vertex]
        MaxtrixSolve[:,1:,:]=np.transpose(A,(0,2,1))
        BCS=lin.solve(MaxtrixSolve,ArraySolve)
        ArrayPhi=argdict[arg][Vertex]
        return np.sum(ArrayPhi*BCS,-1)
    
    def Deriv(self,arg,monoaxe,Coord=None,h=None,argdict=None):
        if argdict is None:
            argdict=self.argdict

        if Coord is None:
            All=True
            Coord=range(self.co.shape[0])
        else:
            All=False
        # print(Coord)
        Phi=argdict[arg][Coord]
        axe=NamesDiff[monoaxe]
        if h is None:
            h=pct*self.dX0[axe]
        dx=np.zeros_like(self.co[Coord])
        dx[:,axe]=h
        CoordPlus=self.co[Coord]+dx
        CoordMoins=self.co[Coord]-dx
        SimplexPlus = self.trig.find_simplex(CoordPlus)
        OutsidePlus=np.where(SimplexPlus<0)



        SimplexMoins = self.trig.find_simplex(CoordMoins)
        OutsideMoins=np.where(SimplexMoins<0)
        if monoaxe == "dt" and self.time_dir:
            CoefMoins=np.ones_like(Phi)
            CoefPlus=np.zeros_like(Phi)

        else:
            CoefPlus,CoefMoins=0.5*np.ones_like(Phi),0.5*np.ones_like(Phi)
            CoefPlus[OutsidePlus]=0
            CoefPlus[OutsideMoins]=1
        CoefMoins[OutsidePlus]=1
        CoefMoins[OutsideMoins]=0

        Phip=self.Interp(arg,CoordPlus,argdict=argdict)
        Phim=self.Interp(arg,CoordMoins,argdict=argdict)
        dPhidaxe=CoefPlus*(Phip-Phi)/h+CoefMoins*(Phi-Phim)/h
        name=f"{arg}_{monoaxe}"
        # print(name)
        if name not in argdict.keys() and All:
            # print("added")
            argdict[name]=dPhidaxe
        return dPhidaxe

    def DerivRecur(self,arg,axe:str,Coord=None,h=None,argdictBase=None,argdictModif=None):
        # print(argdict)
        argdict = self.argdict if argdictBase is None else argdictBase
        # print(arg,axe)
        if f"{arg}_{axe}" in argdict and argdictModif is None:
            # print("Already Done")
            return argdict[f"{arg}_{axe}"][Coord]
        elif argdictModif is not None and f"{arg}_{axe}" in argdictModif:
            return argdictModif[f"{arg}_{axe}"][Coord]
        elif axe in NamesDiff and argdictModif is None:
            return self.Deriv(arg,axe,Coord=Coord,h=h,argdict=argdict)
        elif axe in NamesDiff:
            return self.Deriv(arg,axe,Coord=Coord,h=h,argdict=argdictModif)
        axeList=axe.split("_")
        axe0=axeList[-1]
        newaxe=""
        for naxe in axeList[:-1]:
            newaxe+=naxe
            newaxe+="_"
        newaxe=newaxe[:-1]
        if argdictModif is None:

            self.DerivRecur(arg,newaxe,Coord=Coord,h=h,argdictBase=argdict) #DerivPrior=
            return self.Deriv(f"{arg}_{newaxe}",axe0,Coord=Coord,h=h,argdict=argdict)
        
        else:
            ModifDerivPhi=self.DerivRecur(arg,newaxe,Coord=Coord,h=h,argdictBase=argdict,argdictModif=argdictModif)
            argdictModif[f"{arg}_{newaxe}"]= np.copy(argdictBase[f"{arg}_{newaxe}"])
            argdictModif[f"{arg}_{newaxe}"][Coord]=ModifDerivPhi
            return self.Deriv(f"{arg}_{newaxe}",axe0,Coord=Coord,h=h,argdict=argdictModif)




    
    def PlotGradient2D(self,ax:plt.Axes,arg):
        dphidt = self.argdict[f"{arg}_dt"]
        dphidx = self.argdict[f"{arg}_dx"]
        nope=np.zeros_like(dphidt)
        
        T=self.co[:,0]
        X=self.co[:,1]
        Phi=self.argdict[arg]
        ax.quiver(T,X,Phi,dphidt/100,dphidx/100,nope)
    
    def EvaluateF(self,argdictM=None):
        if argdictM is None:
            argdictM=self.argdict
        argFeval = {arg: argdictM[arg] for arg in self.argF}
        return self.F(**argFeval)
    
    def PPV(self,ind):
        tri=self.trig.simplices

        # print(tri)
        test=[]
        for i in range(tri.shape[0]):
            test.extend(tri[i,:] for indu in ind if indu in tri[i,:])
        # print(test)
        test=np.array(test)
        test=np.reshape(test,test.size)
        test=np.unique(test)
        return test
    
    def GenArgDict(self,Coord=None,h=None,argdict=None):
        for arg in self.argF:
            axeList=arg.split("_")
            if len(axeList)>1:
                newaxe=""
                for naxe in axeList[1:]:
                    newaxe+=naxe
                    newaxe+="_"
                newaxe=newaxe[:-1]
                self.DerivRecur("phi",newaxe,Coord=Coord,h=h,argdictBase=argdict)



    def DerivF(self,co,h=None):
        if h is None:
            h=pct*self.argdict["phi"][co]
            if h ==0.:
                h=1e-6
            # print(h)
        F0=self.EvaluateF(self.argdict)
        modifPhi=np.copy(self.argdict["phi"])
        modifPhi[co]=modifPhi[co]+h
        argdictP = {"phi": modifPhi}
        # print(argdictP["phi"][co],self.argdict["phi"][co])

        degList=[]
        for arg in self.argF:
            axeList=arg.split("_")
            # print(axeList)
            degList.append(len(axeList)-1)
        degList=np.array(degList)
        degList=np.unique(degList)
        # print(degList)
        degList=degList[degList>0]
        CoList={}
        for deg in degList:
            Coord=co
            for _ in range(deg):
                
                Coord=self.PPV(Coord)
                # print(deg,Coord)
            # Coord=[i for i in Coord if i not in g.bound]
            # print(co,Coord)
            CoList[deg]=Coord
        for arg in self.argF:
            axeList=arg.split("_")
            deg=len(axeList)-1
            if deg >0:  
                Coord=CoList[deg]
                newaxe=""
                for naxe in axeList[1:]:
                    newaxe+=naxe
                    newaxe+="_"
                newaxe=newaxe[:-1]
                # print("\n")
                # print(f"phi_{newaxe}")
                # print(Coord)
                # print(self.argdict[f"phi_{newaxe}"][Coord])
                DphiAxe=self.DerivRecur("phi",newaxe,Coord=Coord,argdictBase=self.argdict,argdictModif=argdictP)
                
                DphiAxeTot=np.copy(self.argdict[f"phi_{newaxe}"])
                
                DphiAxeTot[Coord]=DphiAxe
                # print(DphiAxe)
                
                # print(DphiAxe==self.argdict[f"phi_{newaxe}"][Coord])
                 
                
                argdictP[f"phi_{newaxe}"]=DphiAxeTot
                
            # self.DerivRecur("phi",newaxe,Coord,argdict=argdictM)
        # print(argdictP,self.argdict)
        return  (self.EvaluateF(argdictP)-F0)/h
     

    def JacobF(self,h=None):
        n=self.co.shape[0]
        signal=[i for i in range(n) if i not in self.bound ]
        n_signal=len(signal)
        # print(n_signal)
        J=np.zeros((n_signal,n_signal))
        for i,co in enumerate(signal):
            co=np.array([co])
            # print(co)
            Deriv=self.DerivF(co,h=h)
            # print(Deriv.shape)
            Deriv=self.BoundPurge(Deriv,range(n))
            J[:,i]=Deriv
        # J=np.array([self.BoundPurge(self.DerivF(np.array([co]),h=h),range(n)) for co in signal])
        return J

    def NewtonBroy(self,h=None,NbEtape=50,Precision=1e-2):
        # sourcery skip: hoist-statement-from-if
        # if Phi0 is function :
        #     self.SetValueInit(Phi0,"phi")
        # elif Phi0 is np.ndarray:
        #     self.argdict["phi"]=Phi0
        # else:
        #     print("Pas le bon type de Phi0")
        #     return None
        
        self.GenArgDict()
        F0=self.EvaluateF()

        Fk=self.BoundPurge(F0,range(self.co.shape[0]))
        if lin.norm(Fk)<Precision:
            print("Déja bon")
            return self.argdict
        
        # print(J)
        i=0
        while lin.norm(Fk)> Precision and i<NbEtape:
            print(f"Etape {i}")
            if i==0:
                print("Before J")
                J=self.JacobF(h=h)
                print("After J and Before Inv")
                Jinv=lin.inv(J)
                print("After Inv")
                # print(i)
                
            else:
                # J=self.JacobF(h=h)
                # Jinv=lin.inv(J)
                Jinv=Jinv+((DeltaPhi-Jinv@DeltaF)/(DeltaPhi.T@Jinv@DeltaF))@(DeltaPhi.T@Jinv)
            print("Jacob Done")
            
            # print(Jinv)
            Phik=self.argdict["phi"]
            reduced_Phik=np.array(self.BoundPurge(Phik,range(self.co.shape[0])))
            # print(Jinv@Fk)
            reduced_Phik1=-Jinv@Fk+reduced_Phik
            Phik1=np.copy(Phik)
            Phik1[self.signal]=reduced_Phik1
            # print(Phik1)
            # Fk=g.EvaluateF()
            # Fk=np.array(self.BoundPurge(Fk,range(self.co.shape[0])))
            self.argdict={"phi":Phik1}
            self.GenArgDict()
            Fk1=self.EvaluateF()
            Fk1=np.array(self.BoundPurge(Fk1,range(self.co.shape[0])))
            DeltaPhi=reduced_Phik1-reduced_Phik
            DeltaPhi:np.ndarray
            DeltaF=Fk1-Fk
            DeltaF:np.ndarray
            i+=1
            Fk=Fk1
            print("\n")
        return self.argdict
            
            




        







        


t0=ti.time()
plt.figure()
ax=plt.axes(projection='3d')

NbPoint=np.array([20,20])
dX0=np.array([0.05,0.05])
g=Grid(lambda phi_dx_dx,phi_dt_dt,phi: phi_dx_dx-phi_dt_dt+phi,time_dir=True)
g.GeneGridUniEuclid(NbPoint,dX0)
print(g.bound)
# print(g.co)
# f=lambda x: 1 if x[0] ==0. else 0#np.exp(x[0])#1 if x[0]<=0.6 else 0#np.cos(x[0]*0.1-x[1]*0.1) #np.real(np.exp(1j*(-x[0]*1+x[1]*1)))#
def f(x):
    return np.exp(-((x[1]-NbPoint[1]*dX0[1]/2)/dX0[1])**2) if x[0]<=0.3 else 0# np.exp(-((x[1]-0.3)/0.1)**2) 
    
g.SetValueInit(f,"phi")
g.TrigGrid()
g.GenArgDict()
# print(g.EvaluateF())
# ppv=g.PPV([18])
# print(ppv)
# ppv_purge = [i for i in ppv if i not in g.bound]
# print(ppv_purge)

# # print(g.DerivF([18],h=1))
# J=g.JacobF(h=1)
# print(J)

g.PlotPhi2DScalar(ax,"phi",cmap="winter")


print(g.NewtonBroy(h=1e-5,NbEtape=10))
print(ti.time()-t0)
# x=np.array([np.array([0,0]),np.array([5,5])])
# print(g.Interp("phi",x),f(x))
# g.Deriv("phi","dt")
# g.Deriv("phi","dx")
# g.DerivRecur("phi","dt")
# g.DerivRecur("phi","dx")
# g.DerivRecur("phi","dx_dt_dx_dt")
# print(g.argdict.keys())
# # print(g.argdict)
plt.figure()
ax=plt.axes(projection='3d')
g.PlotPhi2DScalar(ax,"phi",cmap="hot")
# plt.figure()
# ax=plt.axes(projection='3d')
# g.PlotPhi2DScalar(ax,"phi_dx",cmap="hot")


# xrange=dX0[0]*np.linspace(0,NbPoint[0]-1,NbPoint[0])
# print(xrange,g.co[:,0])
# yrange=dX0[1]*np.linspace(0,NbPoint[1]-1,NbPoint[0])
# xmesh,ymesh=np.meshgrid(xrange,yrange)
# mesh_interp=lambda x,y:g.Interp("phi",np.array([[x,y]]))
# Result=np.zeros_like(xmesh)
# for i in range(xmesh.shape[0]):
#     for j in range(xmesh.shape[1]):
#         print(xmesh[i,j],ymesh[i,j])
#         Result[i,j]=mesh_interp(xmesh[i,j],ymesh[i,j])
# ax.plot_surface(xmesh,ymesh,Result,cmap="hot")

# print(g.EvaluateF())

# g.PlotGradient2D(ax,"phi")

# c0=np.array([50])
# test1=g.PPV(c0)
# test2=g.PPV(test1)
# print(test1,test2)
plt.show()


