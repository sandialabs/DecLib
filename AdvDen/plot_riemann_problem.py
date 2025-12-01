import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import matplotlib

font = {'size':22}
matplotlib.rc('font', **font)

def multiplot1D(plotname, coords, vardats, labels, ylabel):
    plt.figure(figsize=(20,16))
    for vardat,coord,label in zip(vardats,coords,labels):
        if label == 'exact':
            plt.plot(coord, vardat, label=label, linewidth=3)
        else:
            plt.scatter(coord, vardat, label=label)
    plt.xlabel('x')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig('RPplots/' + plotname + '.png')
    plt.close('all')

def multiplot1D_raw(plotname, vardats, labels, ylabel):
    plt.figure(figsize=(20,16))
    for vardat,label in zip(vardats,labels):
        plt.plot(vardat, label=label, linewidth=3)
    plt.xlabel('x')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig('RPplots/' + plotname + '.png')
    plt.close('all')


def generate_plots(rpexact, ic, time, pname, namelist):
    denslist = []
    plist = []
    ulist = []
    elist = []
    etalist = []
    Slist = []
    Tlist = []
    Pilist = []
    names = []

    coordslist = []

    denslist.append(np.array(rpexact[ic + '/' + time + '/density']))
    plist.append(np.array(rpexact[ic + '/' + time + '/pressure']))
    ulist.append(np.array(rpexact[ic + '/' + time + '/velocity']))
    elist.append(np.array(rpexact[ic + '/' + time + '/energy']))
    etalist.append(np.array(rpexact[ic + '/' + time + '/specific_entropy']))
    Slist.append(np.array(rpexact[ic + '/' + time + '/entropy_density']))
    Tlist.append(np.array(rpexact[ic + '/' + time + '/temperature']))
    coordslist.append(np.linspace(-0.5,0.5,2000))
    names.append('exact')

#ADD THIS BACK IN, EVENTUALLY!
    # cons = h5py.File('cons/' + problem + "/" + namelist[-1] + "/sim.h5", 'r')
    # coordslist.append(np.array(cons['mesh/primal/vertex_locs']))
    # denslist.append(np.array(cons['diag/dens0'][-1,:,0]))
    # elist.append(np.array(cons['diag/inte'][-1,:]))
    # plist.append(np.array(cons['diag/p'][-1,:]))
    # ulist.append(np.array(cons['diag/u'][-1,:]))
    # Tlist.append(np.array(cons['diag/T'][-1,:]))
    # Slist.append(np.array(cons['diag/dens0'][-1,:,1]))
    # etalist.append(np.array(cons['diag/scalar0'][-1,:,1]))
    # names.append('cons')

    for model,nx,eps_type,eps_coeff,label in namelist:
        dirname = model + '/' + ic + '/' + str(nx) + '/' + eps_type + '/' + 'eps=' + str(eps_coeff)
        rp = h5py.File(dirname + "/sim.h5", 'r')

        coordslist.append(np.array(rp['mesh/primal/vertex_locs']))
        denslist.append(np.array(rp['diag/dens0'][-1,:,0]))
        elist.append(np.array(rp['diag/inte'][-1,:]))
        plist.append(np.array(rp['diag/p'][-1,:]))
        ulist.append(np.array(rp['diag/u'][-1,:]))
        Tlist.append(np.array(rp['diag/T'][-1,:]))
        Pilist.append(np.array(rp['diag/Pi'][-1,:]))
        Slist.append(np.array(rp['diag/dens0'][-1,:,1]))
        etalist.append(np.array(rp['diag/scalar0'][-1,:,1]))

        names.append(label)

    trimmednames = names.copy()
    trimmednames.remove('exact')

    multiplot1D(ic + '-density-' + pname, coordslist, denslist, names, 'density')
    multiplot1D(ic + '-pressure-' + pname, coordslist, plist, names, 'pressure')
    multiplot1D(ic + '-velocity-' + pname, coordslist, ulist, names, 'velocity')
    multiplot1D(ic + '-energy-' + pname, coordslist, elist, names, 'energy')
    multiplot1D(ic + '-eta-' + pname, coordslist, etalist, names, 'eta')
    multiplot1D(ic + '-S-' + pname, coordslist, Slist, names, 'S')
    multiplot1D(ic + '-T-' + pname, coordslist, Tlist, names, 'T')
    multiplot1D_raw(ic + '-Pi-' + pname, Pilist, trimmednames, 'Pi')

rpexact = h5py.File("RPplots/RP.h5", 'r')

#generate_plots(rpexact, "RP1", "0.2", 'eps', [('mlp', 2000, "const", 0.0, "0.0"), ('mlp', 2000, "const", 0.0002, "0.0002"), ('mlp', 2000, "const", 0.0001, "0.0001"), ('mlp', 2000, "const", 0.00005, "0.00005"), ('mlp', 2000, "const", 0.000025, "0.000025"), ('mlp', 2000, "const", 0.0000125, "0.0000125")])
#generate_plots(rpexact, "RP1", "0.2", 'N', [('mlp', 100, "const", 0.004, "100"), ('mlp', 500, "const", 0.0008, "500"), ('mlp', 1000, "const", 0.0004, "1000"),('mlp', 2000, "const", 0.0002, "2000")])
#generate_plots(rpexact, "RP1", "0.2", 'epstype2000', [('mlp', 2000, "const", 0.0002, "const"), ('mlp', 2000, "minbee", 0.5, "minbee"), ('mlp', 2000, "minbee_rho", 0.5, "minbee rho")])
#generate_plots(rpexact, "RP1", "0.2", 'epstype500', [('mlp', 500, "const", 0.0008, "const"), ('mlp', 500, "minbee", 0.5, "minbee"), ('mlp', 500, "minbee_rho", 0.5, "minbee rho")])
#generate_plots(rpexact, "RP1", "0.2", 'epstype100', [('mlp', 100, "const", 0.004, "const"), ('mlp', 100, "minbee", 0.5, "minbee"), ('mlp', 100, "minbee_rho", 0.5, "minbee rho")])


experiment_list = []
experiment_list.append(("ModifiedSod", "0.2"))
experiment_list.append(("StreamCollision", "0.8"))
experiment_list.append(("RP3", "0.15"))
experiment_list.append(("RP2", "0.035"))
experiment_list.append(("ToroTest3", "0.012"))
experiment_list.append(("ToroTest4", "0.035"))
experiment_list.append(("RP1", "0.2"))
experiment_list.append(("SlowShock", "2."))


experiment_list.append(("PeakProblem", "3.9e-3"))

#These are all failing still
experiment_list.append(("LeBlanc", "0.5"))
experiment_list.append(("StationaryContact", "0.012"))


for ic, T in experiment_list:
    print(ic,"at",T)
    generate_plots(rpexact, ic, T, 'N', [('mlp', 100, "minbee", 0.5, "100"), ('mlp', 500, "minbee", 0.5, "500"), ('mlp', 1000, "minbee", 0.5, "1000"),('mlp', 2000, "minbee", 0.5, "2000")])
    generate_plots(rpexact, ic, T, 'epstype100', [('mlp', 100, "minbee", 0.5, "minbee"), ('mlp', 100, "minbee_rho", 0.5, "minbee rho")])
    generate_plots(rpexact, ic, T, 'epstype500', [('mlp', 500, "minbee", 0.5, "minbee"), ('mlp', 500, "minbee_rho", 0.5, "minbee rho")])
    generate_plots(rpexact, ic, T, 'epstype2000', [('mlp', 2000, "minbee", 0.5, "minbee"), ('mlp', 2000, "minbee_rho", 0.5, "minbee rho")])


    # if params['init_cond'] == 'RCVCR': return RCVCR(params)
    # if params['init_cond'] == 'VaccumExpansionRight': return VaccumExpansionRight(params)
    # if params['init_cond'] == 'VaccumExpansionLeft': return VaccumExpansionLeft(params)
