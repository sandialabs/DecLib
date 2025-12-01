import matplotlib.pyplot as plt
import numpy as np
import h5py
from dynamics_mlp import AdvDensMLPStatistics, AdvDensMLPDiagnostics, AdvDensMLPDynamics
from dynamics_v import AdvDensVStatistics, AdvDensVDiagnostics, AdvDensVDynamics
from dynamics_advection import AdvectionStatistics, AdvectionDiagnostics, AdvectionDynamics
from dynamics_cons import AdvDensConsStatistics, AdvDensConsDiagnostics, AdvDensConsDynamics

def plot_stat(statname, data):
    plt.figure(figsize=(10,8))
    plt.plot( (data - data[0])/data[0]*100. )
    plt.xlabel('Nsteps')
    plt.ylabel('Fractional Change in ' + statname)
    plt.tight_layout()
    plt.savefig(statname + '.png')

def plot_rawstat(statname, data):
    plt.figure(figsize=(10,8))
    plt.plot(data)
    plt.xlabel('Nsteps')
    plt.ylabel(statname)
    plt.tight_layout()
    plt.savefig(statname + '.raw.png')

def plotvar_scalar1D(plotname, coords, vardat, i):
    plt.figure(figsize=(10,8))
    plt.plot(coords, vardat)
    plt.xlabel('x')
    plt.ylabel(plotname)
    plt.tight_layout()
    plt.savefig(plotname + '.' + str(i) + '.png')
    plt.close('all')

def plotvar_scalar1D_raw(plotname, vardat, i):
    plt.figure(figsize=(10,8))
    plt.plot(vardat)
    plt.xlabel('x')
    plt.ylabel(plotname)
    plt.tight_layout()
    plt.savefig(plotname + '.' + str(i) + '.png')
    plt.close('all')


def plot_model(params):

    oldthermo = params['thermo']
    params['thermo'] = 'EmptyThermo'

    if params['model'] == 'mlp':
        dyn = AdvDensMLPDynamics(None, params, None, construct=False)
        diag = AdvDensMLPDiagnostics(dyn, construct=False)
        stats = AdvDensMLPStatistics(dyn, None, construct=False)
    if params['model'] == 'v':
        dyn = AdvDensVDynamics(None, params, None, construct=False)
        diag = AdvDensVDiagnostics(dyn, construct=False)
        stats = AdvDensVStatistics(dyn, None, construct=False)
    if params['model'] == 'advection':
        dyn = AdvectionDynamics(None, params, None, construct=False)
        diag = AdvectionDiagnostics(dyn, construct=False)
        stats = AdvectionStatistics(dyn, None, construct=False)
    if params['model'] == 'cons':
        dyn = AdvDensConsDynamics(None, params, None, construct=False)
        diag = AdvDensConsDiagnostics(dyn, construct=False)
        stats = AdvDensConsStatistics(dyn, None, construct=False)

    f = h5py.File('sim.h5', 'r')

    denslist = dyn.denslist
    energylist = stats.energylist
    diaglist = diag.diaglist

    stats = f['stats']

    print('plotting stats')
    for l,energy in enumerate(energylist):
        plot_stat(energy, stats['energies'][0,:,l])
        plot_rawstat(energy, stats['energies'][0,:,l])

    for l,dens in enumerate(denslist):
        plot_stat(dens + '_total', stats['dens_total'][0,:,l])
        if 'dens_max' in stats:
            plot_rawstat(dens + '_max', stats['dens_max'][0,:,l])
        if 'dens_min' in stats:
            plot_rawstat(dens + '_min', stats['dens_min'][0,:,l])

    if 'M_total' in stats:
        plot_stat('M_total', stats['M_total'][0,:])

    if 'E_total' in stats:
        plot_stat('E_total', stats['E_total'][0,:])

    #SIGNS ARE BACKWARDS BUT THIS WORKS...
    if 'mass_fluxes' in stats:
        mflux = stats['mass_fluxes'][0,:,:]
        cumulative_mflux = np.cumsum(mflux, axis=0)
        for l,dens in enumerate(denslist):
            plot_stat(dens + '_adj_total', stats['dens_total'][0,:,l] + cumulative_mflux[:,l])

    if 'energy_flux' in stats:
        eflux = stats['energy_flux'][0,:]
        cumulative_eflux = np.cumsum(eflux, axis=0)
        plot_stat('TE_adj', stats['energies'][0,:,energylist.index('TE')] + cumulative_eflux)

    if 'momentum_flux' in stats:
        Mflux = stats['momentum_flux'][0,:]
        cumulative_Mflux = np.cumsum(Mflux, axis=0)
        plot_stat('M_adj_total', stats['M_total'][0,:] + cumulative_Mflux)

    if 'E_flux' in stats:
        Eflux = stats['E_flux'][0,:]
        cumulative_Eflux = np.cumsum(Eflux, axis=0)
        plot_stat('E_adj_total', stats['E_total'][0,:] + cumulative_Eflux)


    #const = f['const']
    diag = f['diag']
    primal = f['mesh/primal']
    dual = f['mesh/dual']

    pcoords = primal['vertex_locs']
    dcoords = dual['vertex_locs']

    Nlist = range(diag['dens0'].shape[0])

    for i in Nlist:
        print('plotting', i)
        for l,dens in enumerate(denslist):
            plotvar_scalar1D(dens, pcoords[:], diag['dens0'][i,:,l], i)
            plotvar_scalar1D(dens+'_scalar', pcoords[:], diag['scalar0'][i,:,l], i)
        # if 'M0' in diag:
            # plotvar_scalar1D('M', pcoords[:], diag['M0'][i,:], i)
        # if 'v' in diag:
            # plotvar_scalar1D_raw('v', diag['v'][i,:], i)
        # if 'u' in diag:
            # plotvar_scalar1D('u', pcoords[:], diag['u'][i,:], i)
    #ARE ALL DIAGNOSTIC VARS USING PCOORDS?
    #NOPE- V IS BROKEN
        for diagvar in diaglist:
            if diagvar == 'v' or diagvar == 'uflux' or diagvar == 'eps' or diagvar == 'alpha':
                plotvar_scalar1D_raw(diagvar, diag[diagvar][i,:], i)
            else:
                plotvar_scalar1D(diagvar, pcoords[:], diag[diagvar][i,:], i)

    params['thermo'] = oldthermo
