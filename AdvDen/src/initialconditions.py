from math import pi, sin, cos, exp, sqrt, erf
import numpy as np
from scipy.special import expi
#from petsc_operators import create_kvar
from math import pi, sin, exp

#initial conditions
#TSWE: double vortex, inflow, SOME 1D ONES?, GEOSTROPHIC BALANCE?
#CE: various Riemann problems, LOTS MORE HERE, RISING BUBBLE?

class IC():

    def __init__(self, params):
        self.params = params

    def set(self, dyn):
        self.prog_vars = dyn.prog_vars
        self.const_vars = dyn.const_vars
        self.pgeom = dyn.meshes.pgeom
        self.dgeom = dyn.meshes.dgeom
        self.ptopo = dyn.meshes.ptopo
        self.dtopo = dyn.meshes.dtopo
        self.dyn = dyn

        self.densfunclist = []
        self.scaled_densfunclist = []
        self.densbndfunclist = []
        self.scaled_densbndfunclist = []
        for dens in self.dyn.denslist:
            if dens=='rho':
                self.densfunclist.append(self.rhofunc)
                self.scaled_densfunclist.append(self.scaled_rhofunc)
                self.densbndfunclist.append(self.rhobndfunc)
                self.scaled_densbndfunclist.append(self.scaled_rhobndfunc)
            elif dens=='S':
                self.densfunclist.append(self.Sfunc)
                self.scaled_densfunclist.append(self.scaled_Sfunc)
                self.densbndfunclist.append(self.Sbndfunc)
                self.scaled_densbndfunclist.append(self.scaled_Sbndfunc)
            elif dens[:2] == 'TA':
                self.densfunclist.append(self.get_tracer_func(self.params['TA' + dens[2] + '_init_cond']))
                self.scaled_densfunclist.append(self.get_scaled_tracer_func(self.params['TA' + dens[2] + '_init_cond']))
                self.densbndfunclist.append(self.get_tracer_bnd_func(self.params['TA' + dens[2] + '_init_cond']))
                self.scaled_densbndfunclist.append(self.get_scaled_tracer_bnd_func(self.params['TA' + dens[2] + '_init_cond']))
            elif dens[:2] == 'TI':
                self.densfunclist.append(self.get_tracer_func(self.params['TI' + dens[2] + '_init_cond']))
                self.scaled_densfunclist.append(self.get_scaled_tracer_func(self.params['TI' + dens[2] + '_init_cond']))
                self.densbndfunclist.append(self.get_tracer_bnd_func(self.params['TI' + dens[2] + '_init_cond']))
                self.scaled_densbndfunclist.append(self.get_scaled_tracer_bnd_func(self.params['TI' + dens[2] + '_init_cond']))
            else: exit('density function for ' + dens + ' not found!')

        print('created ic')

    def get_tracer_func(self, initcond):
        if initcond == 'gaussian': return self.gaussian_tracer
        if initcond == 'square': return self.square_tracer

    def get_scaled_tracer_func(self, initcond):
        if initcond == 'gaussian': return self.scaled_gaussian_tracer
        if initcond == 'square': return self.scaled_square_tracer

    def get_tracer_bnd_func(self, initcond):
        if initcond == 'gaussian': return self.gaussian_tracer_bnd
        if initcond == 'square': return self.square_tracer_bnd

    def get_scaled_tracer_bnd_func(self, initcond):
        if initcond == 'gaussian': return self.scaled_gaussian_tracer_bnd
        if initcond == 'square': return self.scaled_square_tracer_bnd

    def set_IC(self):

        params = self.params

        self.prog_vars['dens'].set_petsc_vec(self.dgeom, self.densfunclist, formtype='all')

        if 'M' in self.prog_vars:
            self.prog_vars['M'].set_petsc_vec(self.dgeom, [self.Mfunc,], formtype='all')
        if 'v' in self.prog_vars:
            self.prog_vars['v'].set_petsc_vec(self.pgeom, [self.vfunc,], formtype='all', linetype='tangent')

        print('set ic')

    def set_uflux(self, auxvars, t):
        auxvars['uflux'].set_petsc_vec(self.dgeom, [self.ufunc,], formtype='all', t=t)

    def set_bnd_dhdxvars(self, bvars, t, scaledof=None):
        params = self.params
        if 'F' in bvars:
            bvars['F'].set_petsc_vec(self.dgeom, [self.Ffunc,], formtype='B', linetype='normal', t=t)
        if 'uflux' in bvars:
            bvars['uflux'].set_petsc_vec(self.dgeom, [self.ufunc,], formtype='B', t=t)

    def set_bnd_jvars(self, bvars, t, scaledof=None):

        if scaledof is None:
            bvars['dens_e'].set_petsc_vec(self.dgeom, self.densbndfunclist, formtype='B', t=t, force_scalar_avg=True)
        else:
            self.scalefunc = self.densfunclist[scaledof]
            bvars['dens_e'].set_petsc_vec(self.dgeom, self.scaled_densbndfunclist, formtype='B', t=t, force_scalar_avg=True)

#THIS WILL NEED ABILITY TO SET BUNDLE-VALUED 1 AND 2-FORMS!!!
        if 'M_e' in bvars:
            bvars['M_e'].set_petsc_vec(self.dgeom, [self.Mbndfunc,], formtype='B', t=t)

    def set_bnd_vars(self, bvars, t, scaledof=None):

        if 'densbound' in bvars:
            bvars['densbound'].set_petsc_vec(self.dgeom, self.densbndfunclist, formtype='B', t=t, force_scalar_avg=True)

        if 'Mbound' in bvars:
#THIS WILL NEED ABILITY TO SET BUNDLE-VALUED 1 AND 2-FORMS!!!
            bvars['Mbound'].set_petsc_vec(self.dgeom, [self.Mbndfunc,], formtype='B', t=t)

        #print('set bnd')

    def set_bnd_fluxvars(self, bvars, t):
        bvars['dens_flux'].set_petsc_vec(self.dgeom, self.densbndfunclist, formtype='B', t=t, force_scalar_avg=True)
#THIS WILL NEED ABILITY TO SET BUNDLE-VALUED 1 AND 2-FORMS!!!
        bvars['M_flux'].set_petsc_vec(self.dgeom, [self.Mbndfunc,], formtype='B', t=t)


class IC1D(IC):

    def scaled_rhofunc(self, t, x):
        return self.rhofunc(t, x) / self.scalefunc(t, x)

    def scaled_Sfunc(self, t, x):
        return self.Sfunc(t, x) / self.scalefunc(t, x)

    def scaled_gaussian_tracer(self, t, x):
        return self.gaussian_tracer(t, x) / self.scalefunc(t, x)

    def scaled_square_tracer(self, t, x):
        return self.square_tracer(t, x) / self.scalefunc(t, x)

    def gaussian_tracer(self, t, x):
        s = 0.1 * (1. + 0.05 * exp(-((x-self.xc)*(x-self.xc))/(1./9*0.5*0.5*self.Lx*self.Lx)))
        return s * self.rhofunc(t, x)

    def square_tracer(self, t, x):
        #xl = self.xc - self.Lx/2
        #xr = self.xc + self.Lx/2
        #x = xscaled * self.Lx + self.xc - self.Lx/2
        xscaled = (x - self.xc + self.Lx/2.)/self.Lx
        if xscaled>0.33333333333333333 and xscaled<0.6666666666666666666: return 0.1* self.rhofunc(t, x)
        else: return 0

#This defaults to setting bc's equal to initial conditions
#IC's override as needed below
    def rhobndfunc(self, t, x):
        return self.rhofunc(t, x)
    def Sbndfunc(self, t, x):
        return self.Sfunc(t, x)
    def Mbndfunc(self, t, x):
        return self.Mfunc(t, x)
    def gaussian_tracer_bnd(self, t, x):
        return self.gaussian_tracer(t,x)
    def square_tracer_bnd(self, t, x):
        return self.square_tracer(t,x)
    def scaled_rhobndfunc(self, t, x):
        return self.scaled_rhofunc(t,x)
    def scaled_Sbndfunc(self, t, x):
        return self.scaled_Sfunc(t,x)
    def scaled_gaussian_tracer_bnd(self, t, x):
        return self.scaled_gaussian_tracer(t,x)
    def scaled_square_tracer_bnd(self, t, x):
        return self.scaled_square_tracer(t,x)

#CAN WE MAKE THIS JIT-TED?
class RiemannProblem(IC1D):
    def __init__(self, params):
        IC.__init__(self, params)
        self.gamma = 1.4
        self.Cv = 1.0

    def set(self, dyn):
        IC.set(self, dyn)

    def rhofunc(self, t, x):
        params = self.params
        if (x <= self.xc_discont):
            return self.rhol
        else:
            return self.rhor

    def Sfunc(self, t, x):
        params = self.params
        rho = self.rhofunc(t, x)
        if (x <= self.xc_discont):
            p = self.pl
        else:
            p = self.pr
        eta = self.dyn.hamiltonian.thermo.get_eta(rho, p)
        return rho * eta

    def Mfunc(self, t, x):
        params = self.params
        if (x <= self.xc_discont):
            return [self.ul * self.rhol,]
        else:
            return [self.ur * self.rhor,]

    def vfunc(self, t, x):
        params = self.params
        if (x <= self.xc_discont):
            return self.ul
        else:
            return self.ur

	#THIS IS MAYBE WRONG- UNCLEAR WHAT THE CORRECT BC IS HERE...
    def ufunc(self, t, x):
        params = self.params
        if (x <= self.xc_discont):
            return self.ul
        else:
            return self.ur

    def Ffunc(self, t, x):
        params = self.params
        return self.ufunc(t,x) * self.rhofunc(t,x)

#Sod Shock Tube
class RiemannProblem1(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1.0
        self.rhor = 0.125
        self.ur =  0.0
        self.pr = 0.1

#Toro Test 5
#DOES THIS HAVE ANOTHER NAME?
class RiemannProblem2(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0 #-0.2
        self.rhol = 5.99924
        self.ul = 19.5975
        self.pl = 460.894
        self.rhor = 5.99242
        self.ur =  -6.19633
        self.pr = 46.095

#Enfield123 ie VaccumExpansion
class RiemannProblem3(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -2.0
        self.pl = 0.4
        self.rhor = 1.0
        self.ur =  2.0
        self.pr = 0.4

class ModifiedSod(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.75
        self.pl = 1.0
        self.rhor = 0.125
        self.ur =  0.0
        self.pr = 0.1


#rho, u, p

#T = 0.012
class ToroTest3(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1000.0
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 0.1


#T = 0.035
class ToroTest4(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 0.01
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 100.0


#T = 0.75
class VaccumExpansionLeft(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 0.0
        self.ul = 0.0
        self.pl = 0.0
        self.rhor = 1.0
        self.ur =  0.0
        self.pr = 1.0




#T = 0.75
class VaccumExpansionRight(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = 1.0
        self.rhor = 0.0
        self.ur =  0.0
        self.pr = 0.0


#T = 0.75
class RCVCR(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -4.0
        self.pl = 0.4
        self.rhor = 1.0
        self.ur =  4.0
        self.pr = 0.4



#T = 0.8
class StreamCollision(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 2.0
        self.pl = 0.1
        self.rhor = 1.0
        self.ur =  -2.0
        self.pr = 0.1


#T = 0.5
class LeBlanc(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = 0.0
        self.pl = (2. / 3.)*1.e-1
        self.rhor = 1.e-3
        self.ur =  0.0
        self.pr = (2. / 3.)*1.e-10


#T = 3.9e-3
class PeakProblem(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 0.1261192
        self.ul = 8.9047029
        self.pl = 782.92899
        self.rhor = 6.591493
        self.ur =  2.2654207
        self.pr = 3.1544874


#T = 2.
class SlowShock(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 3.857143
        self.ul = -0.810631
        self.pl = 10.33333
        self.rhor = 1.0
        self.ur =  -3.44
        self.pr = 1.0

#T = 0.012
class StationaryContact(RiemannProblem):
    def __init__(self, params):
        RiemannProblem.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.0
        self.Lx = 1.0
        self.xc = 0.0

        self.xc_discont = 0.0
        self.rhol = 1.0
        self.ul = -19.59745
        self.pl = 1.e3
        self.rhor = 1.0
        self.ur = 19.59745
        self.pr = 1.e-2

class DensityWave(IC1D):
    def __init__(self, params):
        IC.__init__(self, params)

#FIX THESE ALSO MAYBE?
        params['Lx'] = 2.0
        params['xc'] = 0.
        self.Lx = 2.0
        self.xc = 0.

        self.gamma = 1.4
        self.Cv = 1.0

        self.a = 0.98
        self.u = 0.1
        self.v = 0.2
        self.p = 20.0

    def set(self, dyn):
        IC.set(self, dyn)

    def rhofunc(self, t, x):
        return 1. + self.a * sin(2. * pi * x) #x+y

    def Sfunc(self, t, x):
        rho = self.rhofunc(t, x)
        eta = self.dyn.hamiltonian.thermo.get_eta(rho, self.p)
        return rho * eta

    def Mfunc(self, t, x):
        rho = self.rhofunc(t, x)
        return [rho * self.u,]

    def vfunc(self, t, x):
        return self.u

	#THIS IS MAYBE WRONG- UNCLEAR WHAT THE CORRECT BC IS HERE...
    def ufunc(self, t, x):
        return self.u

    def Ffunc(self, t, x):
        return self.ufunc(t,x) * self.rhofunc(t,x)


class _Gaussian(IC1D):
    def __init__(self, params):
        IC.__init__(self, params)

        params['Lx'] = 1.0
        params['xc'] = 0.5
        self.Lx = 1.0
        self.xc = 0.5

        params['g'] = 9.80616
        params['H0'] =  750.0
        params['dh'] =  75.0

        params['sigmax'] =  3./40.*params['Lx']
        params['xc1'] = 0.5 * params['Lx']
        params['c'] = 0.05
        params['a'] = 1./3.
        params['D'] = 0.5 * params['Lx']

    def rhofunc(self, t, x):
        params = self.params
        xprime1 = params['Lx'] / (pi * params['sigmax']) * sin(pi / params['Lx'] * (x - params['xc1']))
        return params['H0'] + params['dh'] * exp(-0.5 * (xprime1 * xprime1))

    def Mfunc(self, t, x):
        return [0.0,]

    def vfunc(self, t, x):
        return 0.0

    def Sfunc(self, t, x):
        params = self.params
        h = self.rhofunc(t, x)
        s = params['g'] * (1. + params['c'] * exp(-((x-params['xc'])*(x-params['xc']))/(params['a']*params['a']*params['D']*params['D'])))
        return s * h

class Gaussian(_Gaussian):
    def ufunc(self, t, x):
        return 0.0
    def Ffunc(self, t, x):
        return 0.0

class GaussianBackgroundFlow(Gaussian):
    def Mfunc(self, t, x):
        return [0.07 * self.rhofunc(t, x),]

    def vfunc(self, t, x):
        return 0.07

class GaussianInflow(_Gaussian):
    def ufunc(self, t, x):
        if x<0.5:
            return 0.05
        if x>0.5:
            return 0.1
    def Ffunc(self, t, x):
        return self.ufunc(t,x) * self.rhofunc(t, x)

class GaussianInflowBackgroundFlow(GaussianInflow):
    def Mfunc(self, t, x):
        return [0.07 * self.rhofunc(t, x),]

    def vfunc(self, t, x):
        return 0.07

class _Advection(IC1D):
    def __init__(self, params):
        IC.__init__(self, params)
        params['Lx'] = 1.0
        params['xc'] = 0.5
        self.Lx = 1.0
        self.xc = 0.5

    def rhofunc(self, t, x):
        return 1.0

class UniformAdvection(_Advection):
    def ufunc(self, t, x):
        return 1.0

class SineAdvection(_Advection):
    def ufunc(self, t, x):
        params = self.params
        return sin(2.*pi / params['Lx'] * (x - params['xc']))

def getInitialCondition(params):
    if params['init_cond'] == 'RP1': return RiemannProblem1(params)
    if params['init_cond'] == 'RP2': return RiemannProblem2(params)
    if params['init_cond'] == 'RP3': return RiemannProblem3(params)
    if params['init_cond'] == 'ModifiedSod': return ModifiedSod(params)
    if params['init_cond'] == 'ToroTest3': return ToroTest3(params)
    if params['init_cond'] == 'StationaryContact': return StationaryContact(params)
    if params['init_cond'] == 'SlowShock': return SlowShock(params)
    if params['init_cond'] == 'PeakProblem': return PeakProblem(params)
    if params['init_cond'] == 'LeBlanc': return LeBlanc(params)
    if params['init_cond'] == 'StreamCollision': return StreamCollision(params)
    if params['init_cond'] == 'RCVCR': return RCVCR(params)
    if params['init_cond'] == 'VaccumExpansionRight': return VaccumExpansionRight(params)
    if params['init_cond'] == 'VaccumExpansionLeft': return VaccumExpansionLeft(params)
    if params['init_cond'] == 'ToroTest4': return ToroTest4(params)
    if params['init_cond'] == 'DensityWave': return DensityWave(params)
    if params['init_cond'] == 'gaussian': return Gaussian(params)
    if params['init_cond'] == 'gaussianinflow': return GaussianInflow(params)
    if params['init_cond'] == 'gaussianbackground': return GaussianBackgroundFlow(params)
    if params['init_cond'] == 'gaussianinflowbackground': return GaussianInflowBackgroundFlow(params)
    if params['init_cond'] == 'uniformadvection': return UniformAdvection(params)
    if params['init_cond'] == 'sineadvection': return SineAdvection(params)
