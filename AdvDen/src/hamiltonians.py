
from hamil_v import CompressibleEuler_V, TSWE_V
from hamil_mlp import CompressibleEuler_M, TSWE_M

def getHamiltonian(params, meshes, thermodynamics, construct=True):

    if params['hamiltonian'] == 'ce' and params['model'] in ['mlp','cons']:
        hamiltonian = CompressibleEuler_M(meshes, params, thermodynamics, construct=construct)
        
    if params['hamiltonian'] == 'tswe' and params['model'] in ['mlp','cons']:
        hamiltonian = TSWE_M(meshes, params, construct=construct)
        
    if params['hamiltonian'] == 'ce' and params['model'] == 'v':
        hamiltonian = CompressibleEuler_V(meshes, params, thermodynamics, construct=construct)
        
    if params['hamiltonian'] == 'tswe' and params['model'] == 'v':
        hamiltonian = TSWE_V(meshes, params, construct=construct)
        
    return hamiltonian
