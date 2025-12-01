import numpy as np

class _RKBase():
    def __init__(self, dyn, stats, nstages, compute_bnd_fluxes):
        self.dyn = dyn
        self.stats = stats
        self.compute_bnd_fluxes = compute_bnd_fluxes

        self.tendlist = []
        for i in range(nstages):
            self.tendlist.append({})
            for k,v in dyn.prog_vars.items():
                self.tendlist[i][k] = v.duplicate(v.name + '_tend' + str(i))

        self.tenddict = {}
        for k,v in dyn.prog_vars.items():
            self.tenddict[k] = []
            for i in range(nstages):
                self.tenddict[k].append(self.tendlist[i][k].petsc_vec)

        self.auxstate = {}
        for k,v in dyn.prog_vars.items():
            self.auxstate[k] = v.duplicate(v.name + '_auxstate')

#time steppers
#RK4, SSPRK, KGRK, others?
class RK4(_RKBase):
    def __init__(self, dyn, stats, compute_bnd_fluxes):

        _RKBase.__init__(self, dyn, stats, 4, compute_bnd_fluxes)

        self.alphas = np.array([1./6., 1./3., 1./3., 1./6.])

        print('created ts')

    #There are minus signs here since rhs from dyn actually computes dx/dt + rhs = 0!
    def take_step(self, fluxind, dt, t):
        self.dyn.pre_step()

        #k1
        for k,v in self.dyn.prog_vars.items():
            v.petsc_vec.copy(self.auxstate[k].petsc_vec)
        self.dyn.compute_aux(self.auxstate, t, dt)
        self.dyn.compute_rhs(self.tendlist[0], t, dt)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[0], fluxind, t, dt)

        #k2
        for k,v in self.dyn.prog_vars.items():
            self.auxstate[k].petsc_vec.waxpy(-dt/2., self.tendlist[0][k].petsc_vec, v.petsc_vec)
        self.dyn.compute_aux(self.auxstate, t + dt/2.0, dt/2.0)
        self.dyn.compute_rhs(self.tendlist[1], t + dt/2.0, dt/2.0)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[1], fluxind, t + dt/2., dt)

        #k3
        for k,v in self.dyn.prog_vars.items():
            self.auxstate[k].petsc_vec.waxpy(-dt/2., self.tendlist[1][k].petsc_vec, v.petsc_vec)
        self.dyn.compute_aux(self.auxstate, t + dt/2.0, dt/2.0)
        self.dyn.compute_rhs(self.tendlist[2], t + dt/2.0, dt/2.0)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[2], fluxind, t + dt/2., dt)

        #k4
        for k,v in self.dyn.prog_vars.items():
            self.auxstate[k].petsc_vec.waxpy(-dt, self.tendlist[2][k].petsc_vec, v.petsc_vec)
        self.dyn.compute_aux(self.auxstate, t + dt, dt)
        self.dyn.compute_rhs(self.tendlist[3], t + dt, dt)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[3], fluxind, t + dt, dt)

        for k,v in self.dyn.prog_vars.items():
            self.dyn.prog_vars[k].petsc_vec.maxpy(-self.alphas * dt, self.tenddict[k])
            v.petsc_vec.copy(self.auxstate[k].petsc_vec)

        self.dyn.post_step()

        #print('took step')


class SSPRK3(_RKBase):
    def __init__(self, dyn, stats, compute_bnd_fluxes):

        _RKBase.__init__(self, dyn, stats, 3, compute_bnd_fluxes)

        self.alphas = np.array([1./6., 1./6., 2./3.])

        print('created ts')

    #There are minus signs here since rhs from dyn actually computes dx/dt + rhs = 0!
    def take_step(self, fluxind, dt, t):
        self.dyn.pre_step()

        #k1
        for k,v in self.dyn.prog_vars.items():
            v.petsc_vec.copy(self.auxstate[k].petsc_vec)
        self.dyn.compute_aux(self.auxstate, t, dt)
        self.dyn.compute_rhs(self.tendlist[0], t, dt)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[0], fluxind, t, dt)

        #k2
        for k,v in self.dyn.prog_vars.items():
            self.auxstate[k].petsc_vec.waxpy(-dt, self.tendlist[0][k].petsc_vec, v.petsc_vec)
        self.dyn.compute_aux(self.auxstate, t + dt, dt)
        self.dyn.compute_rhs(self.tendlist[1], t + dt, dt)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[1], fluxind, t, dt)

        #k3
        for k,v in self.dyn.prog_vars.items():
            self.auxstate[k].petsc_vec.waxpy(-dt/4., self.tendlist[0][k].petsc_vec, v.petsc_vec)
            self.auxstate[k].petsc_vec.axpy(-dt/4., self.tendlist[1][k].petsc_vec)
        self.dyn.compute_aux(self.auxstate, t + dt/2., dt)
        self.dyn.compute_rhs(self.tendlist[2], t + dt/2., dt)
        if self.compute_bnd_fluxes:
            self.stats.compute_bnd_fluxes(self.alphas[2], fluxind, t + dt/2., dt)

        for k,v in self.dyn.prog_vars.items():
            self.dyn.prog_vars[k].petsc_vec.maxpy(-self.alphas * dt, self.tenddict[k])
            v.petsc_vec.copy(self.auxstate[k].petsc_vec)

        self.dyn.post_step()

#ADD SOME ADDITIONAL RK TIME STEPPERS, ESPECIALLY KGRK AND TS-OPTIMIZED SPPRK SCHEMES

class EC2SI():
    def __init__(self, dyn, stats, compute_bnd_fluxes):
        self.dyn = dyn
        self.stats = stats
        self.compute_bnd_fluxes = compute_bnd_fluxes

        #need some temporaries here!
        #also various solver parameters

    #There are minus signs here since rhs from dyn actually computes dx/dt + rhs = 0!
    def take_step(self, fluxind, dt, t):
        #do newton solver iteration
            #compute dHdx via AVF
            #compute Jvars
            #compute RHS
            #solve linear system
#        self.dyn.post_step()

        pass

class EC2FP():
    def __init__(self, dyn, stats, compute_bnd_fluxes):
        self.dyn = dyn
        self.stats = stats
        self.compute_bnd_fluxes = compute_bnd_fluxes

        #need some temporaries here!
        #also various solver parameters

    #There are minus signs here since rhs from dyn actually computes dx/dt + rhs = 0!
    def take_step(self, fluxind, dt, t):
        #do fixed point iteration
            #compute dHdx via AVF
            #compute Jvars
            #compute RHS
#        self.dyn.post_step()

        pass

def getTimestepper(ts, dyn, stats, compute_bnd_fluxes):
    if (ts == 'RK4'): return RK4(dyn, stats, compute_bnd_fluxes)
    if (ts == 'SSPRK3'): return SSPRK3(dyn, stats, compute_bnd_fluxes)
    if (ts == 'EC2SI'): return EC2SI(dyn, stats, compute_bnd_fluxes)
    if (ts == 'EC2FP'): return EC2FP(dyn, stats, compute_bnd_fluxes)
