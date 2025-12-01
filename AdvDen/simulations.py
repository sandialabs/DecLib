
def get_simulations(experiment):

    simulations = []

    # if experiment == 'RP1':
    #     simulations.append(['mlp', 'RP1', 100, 2000, 100, 0.0001, 'const', 0.004, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 100, 2000, 100, 0.0001, 'const', 0.004, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 100, 2000, 100, 0.0001, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 100, 2000, 100, 0.0001, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 100, 2000, 100, 0.0001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 100, 2000, 100, 0.0001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 100, 2000, 100, 0.0001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 100, 2000, 100, 0.0001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP1', 500, 10000, 500, 0.00002, 'const', 0.0008, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 500, 10000, 500, 0.00002, 'const', 0.0008, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 500, 10000, 500, 0.00002, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 500, 10000, 500, 0.00002, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 500, 10000, 500, 0.00002, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 500, 10000, 500, 0.00002, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 500, 10000, 500, 0.00002, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 500, 10000, 500, 0.00002, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP1', 1000, 20000, 1000, 0.00001, 'const', 0.0004, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 1000, 20000, 1000, 0.00001, 'const', 0.0004, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 1000, 20000, 1000, 0.00001, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 1000, 20000, 1000, 0.00001, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 1000, 20000, 1000, 0.00001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 1000, 20000, 1000, 0.00001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 1000, 20000, 1000, 0.00001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 1000, 20000, 1000, 0.00001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0002, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0002, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0001, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0001, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.00005, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.00005, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.000025, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.000025, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0000125, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP1', 2000, 40000, 2000, 0.000005, 'const', 0.0000125, 'thermodynamicallycompatible'])
    #
    # if experiment == 'RP2':
    #
    #     simulations.append(['mlp', 'RP2', 100, 700, 35, 0.00005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP2', 100, 700, 35, 0.00005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 100, 700, 35, 0.00005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 100, 700, 35, 0.00005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP2', 500, 3500, 175, 0.00001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP2', 500, 3500, 175, 0.00001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 500, 3500, 175, 0.00001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 500, 3500, 175, 0.00001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP2', 1000, 7000, 350, 0.000005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP2', 1000, 7000, 350, 0.000005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 1000, 7000, 350, 0.000005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 1000, 7000, 350, 0.000005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP2', 2000, 14000, 700, 0.0000025, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP2', 2000, 14000, 700, 0.0000025, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 2000, 14000, 700, 0.0000025, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP2', 2000, 14000, 700, 0.0000025, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    # if experiment == 'RP3':
    #
    #     simulations.append(['mlp', 'RP3', 100, 300, 15, 0.0005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP3', 100, 300, 15, 0.0005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 100, 300, 15, 0.0005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 100, 300, 15, 0.0005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP3', 500, 1500, 75, 0.0001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP3', 500, 1500, 75, 0.0001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 500, 1500, 75, 0.0001, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 500, 1500, 75, 0.0001, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP3', 1000, 3000, 150, 0.00005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP3', 1000, 3000, 150, 0.00005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 1000, 3000, 150, 0.00005, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 1000, 3000, 150, 0.00005, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #
    #     simulations.append(['mlp', 'RP3', 2000, 6000, 300, 0.000025, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['mlp', 'RP3', 2000, 6000, 300, 0.000025, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 2000, 6000, 300, 0.000025, 'minbee', 0.5, 'thermodynamicallycompatible'])
    #     simulations.append(['cons', 'RP3', 2000, 6000, 300, 0.000025, 'minbee_rho', 0.5, 'thermodynamicallycompatible'])


    basesizes = {}

# experiment_list.append(("ToroTest3", "0.012"))
# experiment_list.append(("ToroTest4", "0.035"))
# experiment_list.append(("ModifiedSod", "0.2"))
# experiment_list.append(("StationaryContact", "0.012"))
# experiment_list.append(("SlowShock", "2.0"))
# experiment_list.append(("PeakProblem", "3.9e-3"))
# experiment_list.append(("LeBlanc", "0.5"))
# experiment_list.append(("StreamCollision", "0.8"))

    basesizes['RP2'] = [100, 700, 35, 0.00005] #run
    basesizes['RP3'] = [100, 300, 15, 0.0005] #run
    basesizes['StreamCollision'] = [100, 800, 40, 0.001] #run
    basesizes['ModifiedSod'] = [100, 1000, 50, 0.0002] #run
    basesizes['ToroTest3'] = [100, 3000, 150, 0.000004] #run
    basesizes['RP1'] = [100, 2000, 100, 0.0001] #run
    basesizes['ToroTest4'] = [100, 7000, 350, 0.000005] #run
    basesizes['SlowShock'] = [100, 4000, 200, 0.0005] #to be done

#RP1, RP2, Modified Sod all good
#StreamCollision runs but is pretty bad- issue with regularization I think
#RP3 has anomalous temperature spikes- also regularization issue
#Toro Test 3/4 have bad peak heights and also velocity issues
#SlowShock has issues at left boundary- likely a BC problem?

#absurdly high but possible
#I wonder if this also has invariant domain issues...
#2000 runs will take hours, but this is probably okay
    basesizes['PeakProblem'] = [100, 195000, 9750, 0.00000002] #run
#This runs, and seems to be converging
#There are velocity issues

#FAILING WHEN WAVE HITS THE BOUNDARY!!!!
#THIS IS AN ISSUE WITH BCS, I THINK
#SHOULD PROBABLY BE OUTFLOW-TYPE CONDITIONS? Although this is unclear...
#would be useful to do some comparisons at prior times!
    basesizes['StationaryContact'] = [100, 120000, 600, 0.0000001] #to be done

#failing at the beginning near the shock
#likely an issue with invariant domian violations
#seems to be okay initially with dt=0.0000000025 , but this is 200M time steps for nx=100!
    basesizes['LeBlanc'] = [100, 1000, 50, 0.0005] #failing for 100x finer

    basesize = basesizes[experiment]
    for scalefactor in [1,5,10,20]:
        for model in ['mlp','cons']:
            for eps_type,eps_coeff in [('minbee',0.5),('minbee_rho',0.5)]:
                nx,nsteps,nout,dt = basesize
                nx = nx * scalefactor
                nsteps = nsteps * scalefactor
                nout = nout * scalefactor
                dt = dt / scalefactor
                simulations.append([model, experiment, nx, nsteps, nout, dt, eps_type, eps_coeff, 'thermodynamicallycompatible'])

    return simulations
