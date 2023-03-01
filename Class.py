import numpy as np
import scipy.sparse as sp
import itertools as it
import numpy.linalg as lin
import scipy.interpolate as int
import matplotlib.pyplot as plt
import scipy.spatial as spa
pct=0.01
NamesDiff={"dt":0,"dx":1,"dy":2,"dz":3}
DiffNames={0:"dt",1:"dx",2:"dy",3:"dz"}

class Grid:
    def GeneGridUniEuclid(self,NbPoint,dX0):
        self.dX0=dX0
        self.NbPoint=NbPoint
        self.co=np.zeros((np.product(NbPoint),np.size(dX0)))
        ranges=[range(NbPoint[i]) for i in range(NbPoint.size)]
        for i,xs in enumerate(it.product(*ranges)):
            coor=np.array(xs)*dX0
            self.co[i,:]=coor

    def __init__(self,F,argF) -> None:
        self.F=F
        self.argF=argF


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
        CoefPlus,CoefMoins=0.5*np.ones_like(Phi),0.5*np.ones_like(Phi)

        CoordPlus=self.co[Coord]+dx
        SimplexPlus = self.trig.find_simplex(CoordPlus)
        OutsidePlus=np.where(SimplexPlus<0)
        

        CoordMoins=self.co[Coord]-dx
        SimplexMoins = self.trig.find_simplex(CoordMoins)
        OutsideMoins=np.where(SimplexMoins<0)
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

    def DerivRecur(self,arg,axe:str,Coord=None,h=None,argdict=None):
        # print(argdict)
        if argdict is None:
            argdict=self.argdict

        # print(arg,axe)
        if f"{arg}_{axe}" in argdict:
            # print("Already Done")
            return argdict[f"{arg}_{axe}"][Coord]
        elif axe in NamesDiff:
            return self.Deriv(arg,axe,Coord=Coord,h=h,argdict=argdict)
        axeList=axe.split("_")
        axe0=axeList[-1]
        newaxe=""
        for naxe in axeList[:-1]:
            newaxe+=naxe
            newaxe+="_"
        newaxe=newaxe[:-1]
        self.DerivRecur(arg,newaxe,Coord=Coord,h=h,argdict=argdict) #DerivPrior=
        return self.Deriv(f"{arg}_{newaxe}",axe0,Coord=Coord,h=h,argdict=argdict)



    
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
                self.DerivRecur("phi",newaxe,Coord=Coord,h=h,argdict=argdict)



    def DerivF(self,co,h=None):
        if h is None:
            h=pct*self.argdict["phi"][co]
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
                DphiAxe=self.DerivRecur("phi",newaxe,Coord=Coord,argdict=argdictP)
                
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
        J=np.zeros((n,n))
        for co in range(n):
            co=np.array([co])
            J[co,:]=self.DerivF(co,h=h)
        return J 

    def NewtonBroy(self,h=None,NbEtape=50,Precision=1e-5):
        # if Phi0 is function :
        #     self.SetValueInit(Phi0,"phi")
        # elif Phi0 is np.ndarray:
        #     self.argdict["phi"]=Phi0
        # else:
        #     print("Pas le bon type de Phi0")
        #     return None
        self.GenArgDict()
        F0=self.EvaluateF()
        Fk=F0
        if lin.norm(Fk)<Precision:
            print("Déja bon")
            return self.argdict
        J=self.JacobF(h=h)
        print(J)
        i=0
        while lin.norm(Fk)> Precision and i<NbEtape:
            if i==0:
                Jinv=lin.inv(J)
                
                
            else:
                # J=self.JacobF(h=h)
                # Jinv=lin.inv(J)
                Jinv=Jinv+((DeltaPhi-Jinv@DeltaF)/(DeltaPhi.T@Jinv@DeltaF))@(DeltaPhi.T@Jinv)
            
            print(f"Etape {i}")
            print(Jinv)
            Phik=self.argdict["phi"]
            
            Phik1=-Jinv@Fk+Phik
            # print(Phik1)
            Fk=self.EvaluateF()
            self.argdict={"phi":Phik1}
            self.GenArgDict()
            Fk1=self.EvaluateF()
            DeltaPhi=Phik1-Phik
            DeltaPhi:np.ndarray
            DeltaF=Fk1-Fk
            DeltaF:np.ndarray
            i+=1
        return self.argdict
            
            




        







        



plt.figure()
ax=plt.axes(projection='3d')

NbPoint=np.array([5,5])
dX0=np.array([0.5,0.5])
g=Grid(lambda phi_dt_dt,phi_dx_dx: phi_dt_dt-phi_dx_dx,argF=["phi_dt_dt","phi_dx_dx"])
g.GeneGridUniEuclid(NbPoint,dX0)
# print(g.co)
f=lambda x: x[0]**2#np.real(np.exp(1j*(-x[0]*1+x[1]*1)))#
g.SetValueInit(f,"phi")
g.TrigGrid()
g.GenArgDict()()
print(g.EvaluateF())


# print(g.DerivF([18],h=1))
# print(g.JacobF(h=1))
print(g.NewtonBroy(h=1e-2,NbEtape=5))
# x=np.array([np.array([0,0]),np.array([5,5])])
# print(g.Interp("phi",x),f(x))
# g.Deriv("phi","dt")
# g.Deriv("phi","dx")
# g.DerivRecur("phi","dt")
# g.DerivRecur("phi","dx")
# g.DerivRecur("phi","dx_dt_dx_dt")
# print(g.argdict.keys())
# # print(g.argdict)
g.PlotPhi2DScalar(ax,"phi",cmap="hot")
# print(g.EvaluateF())
# g.PlotGradient2D(ax,"phi")

# c0=np.array([50])
# test1=g.PPV(c0)
# test2=g.PPV(test1)
# print(test1,test2)



plt.show()