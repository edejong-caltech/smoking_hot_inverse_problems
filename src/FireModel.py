import numpy as np
from fenics import *
from mshr import *
from dolfin import *
class FireModel():

    def __init__(self, c=1, D=1, S=[], v=[]):
        self.c    = c # concentration of smoke
        self.D    = D # diffusion coefficient
        self.S    = S # sources and sinks (list)
        self.v    = v # local velocity field
        self.data = None

    
    def setDomain(self,xlim,ylim):
        ''' Determines spatial size of simulation; use this if no data is given. 
        
        xlim: list of length 2 with floats as coordinates
        ylim: list of length 2 with floats as coordinates 
        '''
        N         = 10 # number of points on each axis
        self.xmin = xlim[0]
        self.xmax = xlim[1]
        self.ymin = ylim[0]
        self.ymax = ylim[1]
        self.dx   = (xlim[1]-xlim[0])/N
        self.dy   = (ylim[1]-ylim[0])/N


    def generateRandVecField(self):
        '''Generates a nearly-uniform vector field and stores points in self.v .'''
        
        if not ('xmin' in self.__dict__):
            self.setDomain([0,1],[0,1])

        Vx_basis = np.arange(self.xmin,self.xmax,self.dx)
        Vy_basis = np.arange(self.ymin,self.ymax,self.dy)
        Vx,Vy    = np.meshgrid(Vx_basis,Vy_basis)
        self.v   = self.wind(Vx,Vy)


    @np.vectorize
    def wind(x,y):
        # add stuff to this later - generate distribution of wind
        return np.random.randn(1)/10+1 
   
    def uploadData(self,filename):
        '''Retrieves and parses data at location 'filename'.
        
        filename: string, path and filename of data file '''

        # call parse_data
        pass

    
    def computeSim(self,t):
        '''Generates simulation from current saved parameters. 
        
        t: float, length of time for which simulation should run.'''

        
        D = self.D
        c = self.c
        divisions = [10,10]
        W = 100
        degree = 2
        mesh = Rectangle(-W/2, -D, W/2, 0, divisions[0], divisions[1])
        V = FunctionSpace(mesh, 'Lagrange', degree)
        alpha = 3; beta = 1.2
        u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0)

        alpha = 3; beta = 1.2
        u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                        {'alpha': alpha, 'beta': beta})
        u0.t = 0
        
        def boundary(x, on_boundary):  # define the Dirichlet boundary
            return on_boundary
        bc = DirichletBC(V, u0, boundary)
        u_1 = interpolate(u0, V)
        # or
        u_1 = project(u0, V)
        dt = 0.3      # time step

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(beta - 2 - 2*alpha)

        a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
        L = (u_1 + dt*f)*v*dx

        A = assemble(a)   # assemble only once, before the time stepping

        u = Function(V)   # the unknown at a new time level
        T = 2             # total simulation time
        t = dt

        while t <= T:
            b = assemble(L)
            u0.t = t
            bc.apply(A, b)
            solve(A, u.vector(), b)

            t += dt
            u_1.assign(u)
        u_e = interpolate(u0, V)
        maxdiff = np.abs(u_e.vector().array()-u.vector().array()).max()
        print('Max error, t=%.2f: %-10.3f' % (t, maxdiff))

    def parse_data(self):
        '''Centers, rescales data. Called in uploadData function.'''
        pass

    
    def plotSmoke(self):
        '''After simulation has run, plot results and save as jpg/png TBD.'''
        pass