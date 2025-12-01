from helpers import createMeshes

from DecLib import KForm
from DecLib import VolumeFormRecon, LieDerivativeVForm_MLP, DiamondVForm_MLP, LieDerivativeM
from DecLib import LieDerivativeVForm_V, DiamondVForm_V, InteriorProductV
from DecLib import ADD_MODE

#create mesh

# #READ ALL OF THIS FROM A YAML FILE
params = {}

params['meshtype'] = 'line' #'Triangle' 'square' 'meshzoo' 'line' 'dmsh'
params['xbc'] = 'none' #'none' 'periodic'
params['ybc'] = 'none' #'none' 'periodic'
params['nx'] = 100 #100
params['ny'] = 1 #100

params['meshfile'] = 'meshes/box-triangles/high4/square.1'

params['plotmeshes'] = False

params['dmshtype'] = 'square' #square circ
params['dmshsize'] = 0.12

params['optimize'] = True
params['method'] = 'odt-fixed-point'

#THIS GIVES A MESH WITH NON WELL CENTERED CELLS! Good for testing various Hodge stars...
params['zootype'] = 'rectangle-tri' #disk rectangle-tri
params['zoosizes'] = [5,5]

    # "lloyd": cvt.lloyd,
    # "cvt-diaognal": cvt.lloyd,
    # "cvt-block-diagonal": cvt.block_diagonal,
    # "cvt-full": cvt.full,
    # "cpt-linear-solve": cpt.linear_solve,
    # "cpt-fixed-point": cpt.fixed_point,
    # "cpt-quasi-newton": cpt.quasi_newton,
    # "laplace": laplace,
    # "odt-fixed-point": odt.fixed_point,

params['dualtype'] = 'centroid' #'circumcenter' 'centroid'

params['quadorder'] = 3

params['Lx'] = 1.0
params['Ly'] = 1.0
params['xc'] = 0.5
params['yc'] = 0.5

meshes = createMeshes(params)


#create operators
dens_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom)

V_lie_derivative = LieDerivativeVForm_V(meshes)
V_diamond = DiamondVForm_V(meshes)
q_recon = VolumeFormRecon(meshes.ptopo, meshes.pgeom)
v_interior_product = InteriorProductV(meshes)
        
#M_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom)
#m_lie_derivative = LieDerivativeM(meshes)
#MLP_lie_derivative = LieDerivativeVForm_MLP(meshes)
#MLP_diamond = DiamondVForm_MLP(meshes)


#define needed variables
M = KForm(meshes.dim, meshes.dtopo, meshes.dCTBundle, 'M', create_petsc=True)
dens1 = KForm(meshes.dim, meshes.dtopo, meshes.dRBundle, 'dens1', create_petsc=True)
dens2 = KForm(meshes.dim, meshes.dtopo, meshes.dRBundle, 'dens2', create_petsc=True)
B1 = KForm(0, meshes.ptopo, meshes.pRBundle, 'B1', create_petsc=True)
B2 = KForm(0, meshes.ptopo, meshes.pRBundle, 'B2', create_petsc=True)
velocity =  KForm(0, meshes.ptopo, meshes.pTBundle, 'u', create_petsc=True)
flux =  KForm(meshes.dim-1, meshes.dtopo, meshes.dRBundle, 'F', create_petsc=True)
D = KForm(meshes.dim, meshes.dtopo, meshes.dRBundle, 'D', create_petsc=True)

M.petsc_vec.setRandom()
dens1.petsc_vec.setRandom()
dens2.petsc_vec.setRandom()
B1.petsc_vec.setRandom()
B2.petsc_vec.setRandom()
velocity.petsc_vec.setRandom()
flux.petsc_vec.setRandom()
D.petsc_vec.setRandom()

mrecon = KForm(meshes.dim-1, meshes.dtopo, meshes.dCTBundle, 'M_e', create_petsc=True)
dens1recon = KForm(meshes.dim-1, meshes.dtopo, meshes.dRBundle, 'dens1_e', create_petsc=True)
dens2recon = KForm(meshes.dim-1, meshes.dtopo, meshes.dRBundle, 'dens2_e', create_petsc=True)
qrecon = KForm(meshes.dim-1, meshes.dtopo, meshes.dRBundle, 'q_e', create_petsc=True)

dens1rhs = KForm(meshes.dim, meshes.dtopo, meshes.dRBundle, 'dens1rhs', create_petsc=True)
dens2rhs = KForm(meshes.dim, meshes.dtopo, meshes.dRBundle, 'dens2rhs', create_petsc=True)
Mrhs = KForm(meshes.dim, meshes.dtopo, meshes.dCTBundle, 'Mrhs', create_petsc=True)
vrhs = KForm(1, meshes.ptopo, meshes.pRBundle, 'vrhs', create_petsc=True)

# #check advected quantities for mlp
# dens_recon.apply(dens1, dens1recon, flux)
# MLP_diamond.apply(dens1recon, B1, Mrhs)
# MLP_lie_derivative.apply(dens1recon, velocity, dens1rhs, flux)
# Mrhs.petsc_vec.pointwiseMult(Mrhs.petsc_vec, velocity.petsc_vec)
# dens1rhs.petsc_vec.pointwiseMult(dens1rhs.petsc_vec, B1.petsc_vec)
# energy_change = Mrhs.petsc_vec.sum() + dens1rhs.petsc_vec.sum()
# print('mlp advected quantity change', energy_change)

# #check lie derivative for m
# M_recon.apply(M, mrecon, flux)
# m_lie_derivative.apply(mrecon, velocity, Mrhs, flux)
# Mrhs.petsc_vec.pointwiseMult(Mrhs.petsc_vec, velocity.petsc_vec)
# energy_change = Mrhs.petsc_vec.sum()
# print('mlp lie derivative m change', energy_change)

# #check full mlp
# dens_recon.apply(dens2, dens2recon, flux)
# m_lie_derivative.apply(mrecon, velocity, Mrhs, flux)
# MLP_diamond.apply(dens1recon, B1, Mrhs, mode=ADD_MODE)
# MLP_diamond.apply(dens2recon, B2, Mrhs, mode=ADD_MODE)
# MLP_lie_derivative.apply(dens1recon, velocity, dens1rhs, flux)
# MLP_lie_derivative.apply(dens2recon, velocity, dens2rhs, flux)
# Mrhs.petsc_vec.pointwiseMult(Mrhs.petsc_vec, velocity.petsc_vec)
# dens1rhs.petsc_vec.pointwiseMult(dens1rhs.petsc_vec, B1.petsc_vec)
# dens2rhs.petsc_vec.pointwiseMult(dens2rhs.petsc_vec, B2.petsc_vec)
# energy_change = Mrhs.petsc_vec.sum() + dens1rhs.petsc_vec.sum() + dens2rhs.petsc_vec.sum()
# print('mlp full change', energy_change)

#check advected quantities for v
dens_recon.apply(dens1, dens1recon, flux, scale=D, scaledof=0)
V_diamond.apply(dens1recon, B1, vrhs)
V_lie_derivative.apply(dens1recon, velocity, dens1rhs)

#THESE SHOULD REALLY BE TOPOLOGICAL PAIRINGS!

vrhs.petsc_vec.pointwiseMult(vrhs.petsc_vec, flux.petsc_vec)
dens1rhs.petsc_vec.pointwiseMult(dens1rhs.petsc_vec, B1.petsc_vec)
energy_change = vrhs.petsc_vec.sum() + dens1rhs.petsc_vec.sum()
bnd_flux = 
print('v advected quantity change', energy_change - bnd_flux)

#check interior product for v
qrecon.petsc_vec.setRandom() #EVENTUALLY ACTUALLY TEST PROPER RECON HERE!!!
v_interior_product.apply(qrecon, flux, vrhs)

#THESE SHOULD REALLY BE TOPOLOGICAL PAIRINGS!
vrhs.petsc_vec.pointwiseMult(vrhs.petsc_vec, flux.petsc_vec)
energy_change = vrhs.petsc_vec.sum()
print('v interior product change', energy_change)

#check full v
dens_recon.apply(dens2, dens2recon, flux, scale=D, scaledof=0)
v_interior_product.apply(qrecon, flux, vrhs)
V_diamond.apply(dens1recon, B1, vrhs, mode=ADD_MODE)
V_diamond.apply(dens2recon, B2, vrhs, mode=ADD_MODE)
V_lie_derivative.apply(dens1recon, velocity, dens1rhs)
V_lie_derivative.apply(dens2recon, velocity, dens2rhs)

#THESE SHOULD REALLY BE TOPOLOGICAL PAIRINGS!
vrhs.petsc_vec.pointwiseMult(vrhs.petsc_vec, flux.petsc_vec)
dens1rhs.petsc_vec.pointwiseMult(dens1rhs.petsc_vec, B1.petsc_vec)
dens2rhs.petsc_vec.pointwiseMult(dens2rhs.petsc_vec, B2.petsc_vec)
energy_change = vrhs.petsc_vec.sum() + dens1rhs.petsc_vec.sum() + dens2rhs.petsc_vec.sum()
bnd_flux = 
print('v full change', energy_change - bnd_flux)
