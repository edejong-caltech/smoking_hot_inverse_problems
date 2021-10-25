import numpy as np


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
        pass

    
    def parse_data(self):
        '''Centers, rescales data. Called in uploadData function.'''
        pass

    
    def plotSmoke(self):
        '''After simulation has run, plot results and save as jpg/png TBD.'''
        pass