#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import os
import shapely.geometry as geom
import shutil
from scipy.interpolate import interp1d

'''Scalings'''
# Length scale: mm
# Time scale: hours
# The numerical code is entirely dimensional.

'''Parameters'''

# Length scales (all in mm)
Er = 1.81  # embryo radius
r = .92 # margin radius
d = .12 # margin thickness
offset = .106 #initial downward displacement of EP in mm
l_diff = .1583 # diffusion length

# Time scales
tau = 0.5 # myosin regulation time scale [hrs]
emax = 1/3 # maximal contraction rate [1/hrs]
ve_time = 0.5 # margin viscoelastic time scale [hrs]

# Myosin scale
c0 = 1 #set to 1 (unknown units).

# Tension scale
Ts = 1 #set to 1 (unknown number of Newtons per unit myosin)

# Dimensionless biological parameters (controlling myosin regulation in response to stretching)
alpha = -1.0 # regulation offset
beta = 16.0 # sensitivity of myosin to stretching
zeta = 1/3 # amplitude of myosin variation
lam  = 4.0 # #sensitivity of walkers to tension

# Tissue viscosity scale
mu = 0.1 # 2D tissue viscosity ( in units of [c0*Ts] * hours / mm )

# Hair parameters
fr = 500 #hair friction coefficient
xmin = 0 #-10 #minx for hair (set

####

# Margin visco-elasticity (emerging from walking kernel model, for information only)
E = 1/(lam*emax*ve_time) # = 1.5, margin elasticity ( in units of [c0*Ts] )
nu = 1/(lam*emax) # = 0.75, 1D margin viscosity ( in units of [c0*Ts] * hours )

# Dimensionless quantities are only calculated for theoretical interpretation, not used in calculation


'''Initial conditions and numerical settings'''

# Initialisation parameters
Camp = c0*zeta #initial myosin amplitude (perturbation from equilibrium)
initialT = c0*Ts #c0*Ts #initial tension (uniform along margin)

# Regulation settings
dt = 0.05 #time step for FEM and mesh advection
subd = 10 #factor finer time step for margin timestepping
nrep = 161 #number of FEM time steps
remeshinterval = 10 #remesh after this many FEM timesteps

# general parameters
t = 0 #start time
N = 64 #number of margin segments, choose even
dataDir='test_intact_2019' # output data directory

# numerical parameters
meshres = .2 #mesh resolution in mm
pen = 2000 #penalty for weak enforcement of no slip boundary condition if divergence is non-zero, choose large
THorder = 2 #order of the FE method (choose at least 1. Velocity elements have degree THorder + 1)
tol = 1E-10 #numerical tolerance (coefficient of pressure penalty stabilising term)
figureSize = 8 #figure Size
Lambda = Constant(2000) #penalty for weak enforcement of no slip boundary condition if divergence is non-zero, choose large

# switches
regulate = 1 # if 0 tension has fixed gaussian profile WARNING regulate=0 has not yet been tested
remesh = 1 #remesh WARNING REMESHING RESETS THE LOCUS OF THE HAIR (IF PRESENT)
hair = 0 #hair friction on/off.
echo = 0
output = 1 #flag for saving output. 1 = on, 0 = off
slip = 0 #set = 0 for no slip, = 1 for no stress (with moving boundary). WARNING: slip may be buggy.
epibolyflag = 1 # activate epiboly


######################

'''Save parameters'''

if output:
    #copy code to data folder 
    if os.path.exists(dataDir):
        os.system('rm '+dataDir+'/*')
    else:
        os.makedirs(dataDir)
    os.system('jupyter nbconvert --to script Fenics2019_Intact.ipynb')
    os.system('mv Fenics2019_Intact.py '+dataDir)

if output:
    #prepare export
    file1 = File(dataDir+'/u.pvd')   
    file2 = File(dataDir+'/p.pvd')
    file3 = File(dataDir+'/mesh.pvd')
    fT=open(dataDir+'/T.txt','wb')
    fgd=open(dataDir+'/gd.txt','wb')
    fc=open(dataDir+'/c.txt','wb')
    fl=open(dataDir+'/l.txt','wb')
    fps=open(dataDir+'/ps.txt','wb')
    fmvel=open(dataDir+'/margvel.txt','wb')
    ft=open(dataDir+'/time.txt','wt')

'''Initialisation of margin'''

#Create margin segments
angles = 2 * np.pi / N * (np.arange(N)-0.5) - 3*np.pi/2 #set so that 0th segment is at the anterior end
# [was posterior end]
ps = np.array([r*np.cos(angles),r*np.sin(angles)-offset]).T #define array of initial positions

# Margin velocity
margvel=np.zeros(N)

# Extension rate
gd = np.zeros(N)

# Contribution of divergence
halfDiv = np.zeros(N)

# segment lengths l, cumulative lengths cumt, tangent vectors ta, and tangent angles th
def updatelengths(ps):
    l = np.zeros(N)
    cumt = np.zeros(N+1) #cum length of segments measured from midpoint of segments
    ta = np.zeros((N,2)) #tangent vectors
    th = np.zeros(N+1) #tangent angles
    margin = geom.LinearRing(ps) #define linear ring geometric object from array
    lvec = np.roll(ps,-2)-ps
    for i in range(N):
        l[i] = np.linalg.norm(lvec[i,:])
    lt = (l + np.roll(l,-1))/2
    for i in range(N):
        cumt[i+1] = cumt[i]+lt[i]
        ta[i,:] = lvec[i,:]/l[i]
        th[i] = np.arctan2(ta[i,1],ta[i,0])
        if i > 0 and th[i] < th[i-1] - np.pi: #arbitrary threshold, may cause bug
            th[i] += 2*np.pi 
    th[N] = th[0] + 2*np.pi #make periodic
    return margin, l, lt, cumt, th, ta

margin, l, lt, cumt, th, ta = updatelengths(ps)

# peak tension
def maxTension(t):
    return 2.7*mu*(1+np.tanh(2*(t-.95)/2.9))/2*np.exp(-t/8.3) #time in Francis' units
    
# Gaussian tension profile
def tensions(ps,t):
    c = geom.Polygon(ps).centroid # margin center
    mp = (np.roll(ps,-2)+ps)/2 # midpoints
    T = np.zeros(N+1)
    for i in range(N):
        a = np.arctan2(mp[i,0]-c.x,c.y-mp[i,1])
        T[i] = np.exp(-np.square(a/.698)/2)
    T[N] = T[0] # make periodic
    return maxTension(t)*T

# Scalar Tension
if regulate:
    T = initialT * np.ones(N+1)
else:
    T = tensions(ps,t)
dT = np.zeros(N+1)

#Define initial myosin profile
c = c0+Camp*np.cos(th)
c[N] = c[0]
dc = np.zeros(N+1)

cinterp = interp1d(cumt,c,kind='linear') # Myosin interpolator
Tinterp = interp1d(cumt,T,kind='linear') # Tension interpolator
thinterp = interp1d(cumt,th,kind='linear') # tangent angle interpolator


'''Meshing'''

# Define geometry of embryonic disk
currentradius = Er
embryomesh = Circle(Point(0, 0), currentradius)

# Define margin as mesh contour
meshmarginpoints = [0]*N
for i in range(N):
    meshmarginpoints[i] = Point(ps[i,0],ps[i,1])
marginmesh = Polygon(meshmarginpoints)

#Define subregion "hair" with increased friction
if hair:
    hairmesh = Rectangle(Point(xmin,-d/2),Point(currentradius,d/2))

# Label subdomains for later reference (unmarked = index 0)
embryomesh.set_subdomain(1, marginmesh) #EP is subdomain index 1
if hair:
    embryomesh.set_subdomain(2, hairmesh) #hair is subdomain index 2

# generate mesh
mesh = generate_mesh(embryomesh, 2*currentradius/meshres)

# generate markers for domains
markers = MeshFunction('size_t', mesh, 2, mesh.domains()) #Label subdomains with integers

#Plot initial mesh
print("Number of cells:",mesh.num_cells())
plt.figure(figsize=(figureSize,figureSize))
plot(mesh)
plot(markers)
plt.show()

'''Auxiliary functions for finite element modelling of active stresses, hair and epiboly'''

#function to calculate tension at a point x in the embryo
def tisstens(x):
    "Calculate tension at point x according to Gaussian profile with interpolation. Return active stress tensor"
    xp = geom.Point(x) # cast as shapely object
    dist = margin.distance(xp) # distance from margin
    almarg = margin.project(xp) # distance along margin (absolute)   
    the = thinterp((almarg-l[0]/2) % cumt[N])
    tax = (np.cos(the), np.sin(the))
    # Define stress tensor
    s = np.zeros(4)
    s[0] = tax[0]*tax[0]
    s[1] = tax[0]*tax[1] #off diagonal
    s[2] = tax[0]*tax[1] #off diagonal
    s[3] = tax[1]*tax[1]
    # Scale with scalar tension
    ten = 1.0/np.sqrt(2.0*np.pi*pow(d,2))*exp(-pow(dist,2)/(2.0*d*d))*Tinterp((almarg-l[0]/2) % cumt[N])
    s = ten*s
    return s

#Can now define active stress as a class in terms of the subdomain number
class ActiveStress(UserExpression):
    def __init__(self, markers, restrict_range, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers
        self.restrict_range = restrict_range
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0 and self.restrict_range:
            s = np.zeros(4) # set tension to zero away from the margin if restrict_range=true
        else:
            s = tisstens(x)
        for i in range(4):
            values[i] = s[i] # define active stress at point x
    def value_shape(self):
        return (2,2) #active stress is a 2x2 tensor
        
#Define friction coefficient (for hair)
class FrictionCoeff(UserExpression):
    def __init__(self, markers, fr, hair, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers
        self.fr = fr
        self.hair = hair
    def eval_cell(self, values, x, cell):
        if self.hair and self.markers[cell.index] == 2:
            values[0] = self.fr
        else:
            values[0] = 0
    def value_shape(self):
        return () #scalar  
    
#Define expansion properties    
if epibolyflag:
    #Define divergence profiles
    m1 = Expression('0.5*(1-tanh(2*(pow(x[0]*x[0]+(x[1]-os)*(x[1]-os),0.5)-ue)/we))',degree=2,ue=1.0488,we=0.3956,os=-offset)
    m2 = Expression('-x[1]*0.5*(1-tanh(2*(pow(x[0]*x[0]+(x[1]-os)*(x[1]-os),0.5)-ue)/we))',degree=2,ue=1.0488,we=0.3956,os=-offset)
    m3 = Expression('1-0.5*(1-tanh(2*(pow(x[0]*x[0]+(x[1]-os)*(x[1]-os),0.5)-ue)/we))',degree=2,ue=1.0488,we=0.3956,os=-offset)
    m4 = Expression('-exp(-0.5*pow(pow(x[0]*x[0]+x[1]*x[1],0.5)-er,2)/pow(wb,2))',degree=2,er=Er,wb=0.19)
    m5 = Expression('-exp(-0.5*pow(pow(x[0]*x[0]+(x[1]-os)*(x[1]-os),0.5)-rs,2)/pow(ws,2)-0.5*pow(atan2(x[0],-(x[1]-os)),2)/pow(ts,2))',degree=2,rs=0.8694,ts=0.75,ws=0.1196,os=-offset)
    #project onto mesh (piecewise linear)
    V = FunctionSpace(mesh, "DG", 2) #quadratic elements to mitigate resolution loss during contraction
    div1 = project(m1, V)
    div2 = project(m2, V)
    div3 = project(m3, V)
    div4 = project(m4, V)
    div5 = project(m5, V)
    #allow for extrapolation (to avoid errors during remeshing)
    div1.set_allow_extrapolation(True)
    div2.set_allow_extrapolation(True)
    div3.set_allow_extrapolation(True)
    div4.set_allow_extrapolation(True)
    div5.set_allow_extrapolation(True)
    #Coefficients for epiboly contributions
    coef1 = Constant(0.27/8) # Constant(0.27/8)
    coef2 = Constant(0)
    coef3 = Constant(1.58/8) # Constant(1.58/8)
    coef4 = Constant(1.06/8) # Constant(1.06/8)
    coef5 = Constant(2.26/2.901*(1+np.tanh(2*(t-5.1)/2.0))/2) # Constant(2.26/2.9*(1+np.tanh(2*(t-5.1)/2.0)/2))
    
    
'''Solving the variational problem'''

# Define function spaces, choose Taylor-Hood elements
P2 = VectorElement("Lagrange", mesh.ufl_cell(), THorder+1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), THorder)
TH = P2 * P1
W = FunctionSpace(mesh, TH)
 
# auxiliary function for strong enforcement of Dirichlet BC
def boundary(x, on_boundary):
    return on_boundary

if slip:
    #no shear stress
    pass
else:
    #no slip BC
    if epibolyflag:
        # Determine boundary velocity from integrated area expansion
        # apply weakly and incur penalty with weight lambda
        areaexp = assemble((coef1*div1+coef3*div3+coef4*div4+coef5*div5)*dx(mesh))
        circumf = assemble(Constant(1.0)*ds(mesh))
        un = areaexp/circumf #divergence theorem:  div(u) dx = u.n ds
        UN = Constant(un)
        n = FacetNormal(mesh)
    else:
        # boundary fixed if no epiboly, enforce constraint strongly
        u0 = Constant((0, 0))
        bc = DirichletBC(W.sub(0), u0, boundary)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
Mu = Constant(mu)
TOL = Constant(tol) #pressure penalty term for numerical stability

#Define active stress for variational form    
s = ActiveStress(markers,False,degree=2)    
#Define hair friction as appropriate
if hair:
    fric = FrictionCoeff(markers, fr, hair, degree=0)
else:
    fric = Constant(0.0)  

#Note: bilinear form a is symmetric for computational efficiency
if epibolyflag:
    if slip:
        #no constraint on boundary motion
        a = (Mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u) - TOL*p*q + fric*dot(u,v))*dx
        L = -inner(s, grad(v))*dx - (coef1*div1+coef3*div3+coef4*div4+coef5*div5)*q*dx
    else:
        #enforce no slip BC weakly
        a = (Mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u) - TOL*p*q + fric*dot(u,v))*dx + Lambda*dot(u,v)*ds
        L = -inner(s, grad(v))*dx - (coef1*div1+coef3*div3+coef4*div4+coef5*div5)*q*dx + Lambda*UN*dot(n,v)*ds
else:
    a = (Mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u)- TOL*p*q+fric*dot(u,v))*dx
    L = -inner(s, grad(v))*dx

# Compute solution
w = Function(W)
if (not slip) and (not epibolyflag):
    #enforce Dirichlet BC strongly
    solve(a == L, w, bc)
else:
    solve(a == L, w)

#Loop
for j in range(nrep):

    #Plot mesh
    plt.figure(figsize=(figureSize,figureSize))
    plt.title('t={:.1f}'.format(t))
    plot(mesh)
    plt.show()    

    #Plot Myosin, tension and contraction rate profiles
    plt.figure(figsize=(figureSize,figureSize))
    plt.title('t={:.1f}'.format(t))
    if regulate:
        plt.plot(cumt,c)
    plt.plot(cumt,T/Ts)
    plt.plot(cumt[0:N],gd/emax)
    plt.ylim(-2,2);
    plt.show()    
    
    if epibolyflag:
        #if desired re-calculate margin and remesh
        if remesh and ((j+1) % remeshinterval == 0):
            '''Re-define margin'''
            #create newps by interpolating margin at regular intervals
            newps = np.zeros((N,2))
            for i in range(N):
                newps[i,:] = np.array(margin.interpolate(i*np.sum(l)/N))
            newmargin, newl, newlt, newcumt, newth, newta = updatelengths(newps)
            
            #interpolate myosin and tension to new edge midpoints
            newc = cinterp((newcumt+(newl[0]-l[0])/2) % cumt[N])
            newT = Tinterp((newcumt+(newl[0]-l[0])/2) % cumt[N])
            #the new polygonal contour may differ in length from the original one, so need to ensure periodicity manually
            newc[N] = newc[0]
            newT[N] = newT[0]
        
            #define new interpolators
            newcinterp = interp1d(newcumt,newc,kind='linear') # Myosin interpolator
            newTinterp = interp1d(newcumt,newT,kind='linear') # Tension interpolator
            newthinterp = interp1d(newcumt,newth,kind='linear') # tangent angle interpolator
        
            #update variables
            ps = newps
            margin = newmargin
            l = newl
            lt = newlt
            cumt = newcumt
            th = newth
            ta = newta
            c = newc
            T = newT
            cinterp = newcinterp
            Tinterp = newTinterp
            thinterp = newthinterp 
    
    '''update/determine margin variables. In case of remeshing, calculate using new margin but old mesh'''
    #obtain velocity and pressure fields
    (u, p) = w.split()
    for i in range(N):
        #margin velocity (for information only)
        margvel[i]=np.linalg.norm(u(ps[i,:]))
        #contraction rates
        gd[i] = np.dot(u(ps[(i+1)%N,:])-u(ps[i,:]),ta[i,:])/l[i]
        mp = (ps[(i+1)%N,:]+ps[i,:])/2
        halfDiv[i] = (coef1*div1(mp[0],mp[1])+coef3*div3(mp[0],mp[1])+coef4*div4(mp[0],mp[1])+coef5*div5(mp[0],mp[1]))/2
    
    if output:
        file1 << u
        file2 << p   
        file3 << mesh
        np.savetxt(fmvel, margvel, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        np.savetxt(fT, T, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        np.savetxt(fgd, gd, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        np.savetxt(fc, c, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        np.savetxt(fl, l, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        np.savetxt(fps, ps, fmt='%.4e', delimiter=' ', newline='\n', header='', footer='\n', comments='# ')
        print(t,file=ft)
    
    #Display information
    if echo:
        print("Time: %.3g" % t)
        if epibolyflag and (not slip):
            print("Boundary velocity: %.4g" % un)
        print("Max margin velocity: %.4g" % np.amax(margvel))
        print("Max margin extension rate: %.4g" % np.amax(gd))
        print("Min margin extension rate: %.4g" % np.amin(gd))
        print("Max Tension: %.4g" % np.amax(T))
        print("Min Tension: %.4g" % np.amin(T))
        print("Max Myosin: %.4g" % np.amax(c))
        print("Min Myosin: %.4g" % np.amin(c))
    
    #Plot velocity field
    plt.figure(figsize=(figureSize,figureSize))
    plot(u)
    plt.show()

    '''
    plt.figure(figsize=(figureSize,figureSize))
    plot(p)
    plt.show()
    '''
    
    #update margin locus, myosin and tension
    for i in range(N):
        ps[i,:] += dt * u(ps[i,:])
        
    if regulate:
        for it in range(subd):
            for i in range(N):
                dc[i] = 1/tau*(c0+c0*zeta*np.tanh(alpha-beta*gd[i]/(lam*emax)) - c[i] + l_diff**2 * (lt[i%N]*c[(i-1)%N]+lt[(i-1)%N]*c[(i+1)%N]-(lt[i%N]+lt[(i-1)%N])*c[i])/(1/2*(lt[i%N]+lt[(i-1)%N])*lt[i%N]*lt[(i-1)%N]) )
                dT[i] = c0*Ts/(lam*emax*ve_time)*(gd[i]-halfDiv[i]-emax*np.tanh(lam*(T[i]/c[i]/Ts-1)))
            dc[N] = dc[0]
            dT[N] = dT[0]
            c += dt/subd * dc
            T += dt/subd * dT
    else:
        T = tensions(ps,t)
        
    margin, l, lt, cumt, th, ta = updatelengths(ps)
    cinterp = interp1d(cumt,c,kind='linear') # Myosin interpolator
    Tinterp = interp1d(cumt,T,kind='linear') # Tension interpolator
    thinterp = interp1d(cumt,th,kind='linear') # tangent interpolator
    
    if slip or epibolyflag:
        #Move mesh
        u.vector()[:] *= dt
        ALE.move(mesh,u)
        mesh.bounding_box_tree().build(mesh)
        #Update radius
        currentradius+=dt*un

        '''
        plt.figure(figsize=(figureSize,figureSize))
        plot(mesh)
        plot(markers)
        plt.show()
        '''
        
        if remesh and ((j+1) % remeshinterval == 0):
            '''Redefine mesh'''
            # Define geometry of embryonic disk
            embryomesh = Circle(Point(0, 0), currentradius)
    
            # Define margin as mesh contour
            meshmarginpoints = [0]*N
            for i in range(N):
                meshmarginpoints[i] = Point(ps[i,0],ps[i,1])
            marginmesh = Polygon(meshmarginpoints)

            #Define subregion "hair" with increased friction
            if hair:
                hairmesh = Rectangle(Point(xmin,-d/2),Point(currentradius,d/2))
               
            # Label subdomains for later reference (unmarked = index 0)
            embryomesh.set_subdomain(1, marginmesh) #EP is subdomain index 1
            if hair:
                embryomesh.set_subdomain(2, hairmesh) #hair is subdomain index 2

            # generate mesh
            newmesh = generate_mesh(embryomesh, 2*currentradius/meshres)

            # generate markers for domains
            markers = MeshFunction('size_t', newmesh, 2, newmesh.domains()) #Label subdomains with integers
    
            #Project contractility map onto new mesh
            Vnew = FunctionSpace(newmesh, "DG", 2)
            newdiv1 = project(div1,Vnew)
            newdiv2 = project(div2,Vnew)
            newdiv3 = project(div3,Vnew)
            newdiv4 = project(div4,Vnew)
            newdiv5 = project(div5,Vnew)
            div1 = newdiv1
            div2 = newdiv2
            div3 = newdiv3
            div4 = newdiv4
            div5 = newdiv5
            div1.set_allow_extrapolation(True)
            div2.set_allow_extrapolation(True)
            div3.set_allow_extrapolation(True)
            div4.set_allow_extrapolation(True)
            div5.set_allow_extrapolation(True)
        
            #update active stress and finite element structure
            mesh = newmesh
            s = ActiveStress(markers,False,degree=2)
            if hair:
                fric = FrictionCoeff(markers, fr, hair, degree=0)
            else:
                fric = Constant(0.0)  
            P2 = VectorElement("Lagrange", mesh.ufl_cell(), THorder+1)
            P1 = FiniteElement("Lagrange", mesh.ufl_cell(), THorder)
            TH = P2 * P1
            W = FunctionSpace(mesh, TH)
            (u, p) = TrialFunctions(W)
            (v, q) = TestFunctions(W)
        
        #Clear u and p
        (u, p) = TrialFunctions(W)
        
        #Update margin contraction coefficient (the only time-dependent one)
        coef5 = Constant(2.26/2.901*(1+np.tanh(2*(t-5.1)/2.0))/2) # Constant(2.26/2.9*(1+np.tanh(2*(t-5.1)/2.0))/2)
        
        if epibolyflag and (not slip):
            #update divergence
            areaexp = assemble((coef1*div1+coef3*div3+coef4*div4+coef5*div5)*dx(mesh))
            circumf = assemble(Constant(1.0)*ds(mesh))
            un = areaexp/circumf #divergence theorem:  div(u) dx = u.n ds
            UN = Constant(un)
            n = FacetNormal(mesh)
            a = (Mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u)-(1E-10)*p*q+fric*dot(u,v))*dx + Lambda*dot(u,v)*ds
            L = -inner(s, grad(v))*dx - (coef1*div1+coef3*div3+coef4*div4+coef5*div5)*q*dx + Lambda*UN*dot(n,v)*ds
        if epibolyflag and slip:
            a = (Mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u)-(1E-10)*p*q+fric*dot(u,v))*dx
            L = -inner(s, grad(v))*dx - (coef1*div1+coef3*div3+coef4*div4+coef5*div5)*q*dx

    
    #Solve
    w = Function(W)
    if (not slip) and (not epibolyflag):
        solve(a == L, w, bc)
    else:
        solve(a == L, w)

    #update timestep
    t += dt

if output:    
    fT.close()
    fgd.close()
    fc.close()
    fl.close()
    fps.close()
    fmvel.close()
    ft.close()


# In[ ]:




