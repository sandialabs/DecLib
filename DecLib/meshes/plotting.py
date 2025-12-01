import numpy as np
import matplotlib.pyplot as plt
import matplotlib

kcelltypes = ['I', 'b', 'B']

def plot1Dmesh(topo, geom, quad, name):

    plt.figure(figsize=(10,4))

    vcoords = np.reshape(geom.vertexcoords, topo.nkcells[0])
    minx, maxx = np.min(vcoords), np.max(vcoords)
    xtot = maxx - minx
    dx = xtot/topo.nkcells[1]
    plt.xlim(minx - xtot*0.1, maxx + xtot*0.1 + dx)
    plt.ylim(-.10,0.10)

# vertex locations (nv, dim) and labels
    vcoords = np.reshape(geom.vertexcoords, topo.nkcells[0])
    plt.scatter(vcoords, np.zeros(vcoords.shape[0]), s=1000, marker="|", color='orange')
    voff = topo.kcells_off[0]
    for v in range(topo.nkcells[0]):
        plt.text(vcoords[v], -0.03, str(v))
        kcelltype = topo.petscmesh.getLabelValue('bnd', v+voff)
        plt.text(vcoords[v], -0.06, kcelltypes[kcelltype])


# edge centroids (ne, dim) and labels
    centroids = np.reshape(geom.centroids[1], topo.nkcells[1])
    plt.scatter(centroids, np.zeros(centroids.shape[0]), s=100, marker="x", color='red')
    coff = topo.kcells_off[1]
    for c in range(topo.nkcells[1]):
        plt.text(centroids[c], 0.02, str(c))
        kcelltype = topo.petscmesh.getLabelValue('bnd', c+coff)
        plt.text(centroids[c], -0.06, kcelltypes[kcelltype])

# edge segments (ne, ns, dim)
    for e in range(topo.nkcells[1]):
        segments = np.ravel(geom.edge_segments[e, :geom.num_edge_segments[e], :, 0])
        plt.scatter(segments, np.zeros(segments.shape[0]) + 0.04, s=100, marker="o", color='blue')

# face normals
    dx = xtot/topo.nkcells[1]
    for v in range(topo.nkcells[0]):
        facenormal = geom.face_normals[v,0]
        plt.arrow(vcoords[v], 0.0, facenormal*dx/4., 0.0, length_includes_head=True,
        head_starts_at_zero=True, overhang=0.0,width=0.0005, head_width=6*0.0005,
        head_length=48*0.0005, shape='full', color='k')

# vertex quad pts
    vquad = np.reshape(quad.quadpts[0], topo.nkcells[0])
    plt.scatter(vquad, np.zeros(vquad.shape[0]) + 0.06, s=100, marker="+", color='green')

# edge quad pts + wts + face normals (all per segment)
    for e in range(topo.nkcells[1]):
        equad = np.ravel(quad.quadpts[1][e, :geom.num_edge_segments[e], :, 0])
        plt.scatter(equad, np.zeros(equad.shape[0]) + 0.06, s=100, marker="+", color='black')

    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.close('all')

def plot2Dmesh(topo, geom, quad, name, celltype='all'):
    nv = topo.nkcells[0]
    vcoords = geom.vertexcoords
    plt.figure(figsize=(20,16))

    minx, maxx = np.min(vcoords[:,0]), np.max(vcoords[:,0])
    miny, maxy = np.min(vcoords[:,1]), np.max(vcoords[:,1])
    xtot = maxx - minx
    ytot = maxy - miny
    dx = np.sqrt(xtot*ytot/topo.nkcells[2])

    plt.xlim(minx - xtot*0.1, maxx + xtot*0.1 + 1.3*dx)
    plt.ylim(miny - ytot*0.1, maxy + ytot*0.1 + 1.3*dx)

# vertex locations and labels
    voff = topo.kcells_off[0]
    for v in range(topo.nkcells[0]):
        kcelltype = topo.petscmesh.getLabelValue('bnd', v+voff)
        if celltype=='all' or celltype==kcelltypes[kcelltype]:
            plt.text(vcoords[v,0], vcoords[v,1] - 0.03*xtot, 'v'+str(v))
            plt.scatter(vcoords[v,0], vcoords[v,1], s=100, marker="o", color='orange')

# edge centroids (ne, dim) and labels
# edge segments (ne, ns, dim)
# edge quad pts + wts + edge tangents + face normals (all per segment)


    eoff = topo.kcells_off[1]
    for e in range(topo.nkcells[1]):
        kcelltype = topo.petscmesh.getLabelValue('bnd', e+eoff)
        if celltype=='all' or celltype==kcelltypes[kcelltype]:
            plt.scatter(geom.centroids[1][e,:,0], geom.centroids[1][e,:,1], s=100, marker="x", color='red')


            #PUTS TEXT AT 1ST SEGMENT EDGE CENTROID
            #PROBABLY FINE
            plt.text(geom.centroids[1][e,0,0], geom.centroids[1][e,0,1] - 0.03*xtot, 'e'+str(e))
            segments = geom.edge_segments[e, :geom.num_edge_segments[e], :, :]
            for p in range(geom.num_edge_segments[e]):
                plt.plot(segments[p,:,0], segments[p,:,1], color='black')
                #plt.scatter(, s=30, marker="o", color='blue')

                segment_centroid = geom.centroids[1][e, p, :]
                facenormal = geom.face_normals[e,p,:]
                plt.arrow(segment_centroid[0], segment_centroid[1], facenormal[0]*dx/4., facenormal[1]*dx/4., length_includes_head=True,
                head_starts_at_zero=True, overhang=0.0,width=0.003, head_width=3*0.003,
                head_length=48*0.0005, shape='full', color='green')

                edgetangent = geom.edge_tangents[e,p,:]
                plt.arrow(segment_centroid[0], segment_centroid[1], edgetangent[0]*dx/4., edgetangent[1]*dx/4., length_includes_head=True,
                head_starts_at_zero=True, overhang=0.0,width=0.003, head_width=3*0.003,
                head_length=48*0.0005, shape='full', color='blue')

                equad = quad.quadpts[1][e, p, :, :]
                plt.scatter(equad[:,0], equad[:,1], s=100, marker="P", color='black')



# face centroids
# face triangles (keyed to a given edge)
# face quad pts + wts (all per triangle)
    foff = topo.kcells_off[2]
    for f in range(topo.nkcells[2]):
        kcelltype = topo.petscmesh.getLabelValue('bnd', f+foff)
        if celltype=='all' or celltype==kcelltypes[kcelltype]:
            if topo.is_simplicial:
                plt.scatter(geom.circumcenters[f,0], geom.circumcenters[f,1], s=100, marker="D", color='blue')
            plt.scatter(geom.centroids[2][f,0], geom.centroids[2][f,1], s=100, marker="D", color='purple')
            plt.text(geom.centroids[2][f,0], geom.centroids[2][f,1] - 0.03*xtot, 'f'+str(f))
            for e in range(topo.nEF[f]):
                for p in range(geom.num_edge_segments[e]):
                    facetriangle = geom.face_triangles[f, e, p, :, :]
                    plt.scatter(facetriangle[:,0], facetriangle[:,1], s=10, marker="o", color='orange')
                    fquad = quad.quadpts[2][f, e, p, :, :]
                    plt.scatter(fquad[:,0], fquad[:,1], s=100, marker="2", color='black')

    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.close('all')

#From https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

def _plot_cell(topo, geom, cind, cell_tet, centroid, rough_centroid, name):

    fig = plt.figure(figsize=(20,16))
    #cell tet is f,e,p,4,3
    cell_tet_x = cell_tet[:,:,:,:,0]
    cell_tet_y = cell_tet[:,:,:,:,1]
    cell_tet_z = cell_tet[:,:,:,:,2]

    minx, maxx = np.min(cell_tet_x), np.max(cell_tet_x)
    miny, maxy = np.min(cell_tet_y), np.max(cell_tet_y)
    minz, maxz = np.min(cell_tet_z), np.max(cell_tet_z)
    xtot = maxx - minx
    ytot = maxy - miny
    ztot = maxz - minz

    ax = fig.add_subplot(projection='3d')

    ax.set_xlim3d(minx - xtot*0.5, maxx + xtot*0.5)
    ax.set_ylim3d(miny - ytot*0.5, maxy + ytot*0.5)
    ax.set_zlim3d(minz - ztot*0.5, maxz + ztot*0.5)

    ax.scatter(cell_tet_x[:,:,:,:2], cell_tet_y[:,:,:,:2], cell_tet_z[:,:,:,:2], s=100, marker="o", color='orange')
    ax.scatter(cell_tet_x[:,:,:,2], cell_tet_y[:,:,:,2], cell_tet_z[:,:,:,2], s=100, marker="o", color='green')
    ax.scatter(centroid[0], centroid[1], centroid[2], s=100, marker="D", color='green')
    ax.scatter(rough_centroid[0], rough_centroid[1], rough_centroid[2], s=100, marker="2", color='green')

    foff = topo.kcells_off[2]
    eoff = topo.kcells_off[1]
    for find,f in enumerate(topo.FC[cind, :topo.nFC[cind]]):
        for eind,e in enumerate(topo.EF[f-foff, :topo.nEF[f-foff]]):
            for p in range(geom.num_edge_segments[e-eoff]):
                ax.plot(cell_tet[find,eind,p,:2,0], cell_tet[find,eind,p,:2,1], cell_tet[find,eind,p,:2,2], color='black')

#ADD NORMALS

    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.close('all')



def plot3Dmesh(topo, geom, quad, name, celltype='all'):

    nv = topo.nkcells[0]
    vcoords = geom.vertexcoords
    fig = plt.figure(figsize=(20,16))

    minx, maxx = np.min(vcoords[:,0]), np.max(vcoords[:,0])
    miny, maxy = np.min(vcoords[:,1]), np.max(vcoords[:,1])
    minz, maxz = np.min(vcoords[:,2]), np.max(vcoords[:,2])
    xtot = maxx - minx
    ytot = maxy - miny
    ztot = maxz - minz
    dx = np.sqrt(xtot*ytot*ztot/topo.nkcells[3])

    vstart, vend = topo.get_zerobased_loop_indices(0, celltype)
    estart, eend = topo.get_zerobased_loop_indices(1, celltype)
    fstart, fend = topo.get_zerobased_loop_indices(2, celltype)
    cstart, cend = topo.get_zerobased_loop_indices(3, celltype)

    ax = fig.add_subplot(2, 2, 1, projection='3d')

    ax.set_xlim3d(minx - xtot*0.1, maxx + xtot*0.1 + 1.3*dx)
    ax.set_ylim3d(miny - ytot*0.1, maxy + ytot*0.1 + 1.3*dx)
    ax.set_zlim3d(minz - ztot*0.1, maxz + ztot*0.1 + 1.3*dx)

#vertex locations
    ax.scatter(vcoords[vstart:vend,0], vcoords[vstart:vend,1], vcoords[vstart:vend,2], s=100, marker="o", color='orange')

# edge centroids (ne, dim) and labels
# edge segments (ne, ns, dim)
# edge quad pts + wts + edge tangents (all per segment)
    ax.scatter(quad.quadpts[1][estart:eend, :, :, 0], quad.quadpts[1][estart:eend, :, :, 1], quad.quadpts[1][estart:eend, :, :, 2], s=100, marker="P", color='black')
    for e in range(estart,eend):
        for p in range(geom.num_edge_segments[e]):
            ax.plot(geom.edge_segments[e, p, :, 0], geom.edge_segments[e, p, :, 1], geom.edge_segments[e, p, :, 2], color='black')
            segment_centroid = geom.centroids[1][e, p, :]
            edgetangent = geom.edge_tangents[e, p, :]
            #print(e,p,dx,segment_centroid,edgetangent)
            ax.arrow3D(segment_centroid[0], segment_centroid[1], segment_centroid[2], edgetangent[0]*dx/4., edgetangent[1]*dx/4., edgetangent[2]*dx/4.,
            mutation_scale=20, fc='blue')


# face centroids
# face triangles (keyed to a given edge)
# face quad pts + wts (all per triangle)
# face normals

    ax = fig.add_subplot(2, 2, 2, projection='3d')

    ax.set_xlim3d(minx - xtot*0.1, maxx + xtot*0.1 + 1.3*dx)
    ax.set_ylim3d(miny - ytot*0.1, maxy + ytot*0.1 + 1.3*dx)
    ax.set_zlim3d(minz - ztot*0.1, maxz + ztot*0.1 + 1.3*dx)

    for e in range(estart,eend):
        for p in range(geom.num_edge_segments[e]):
            ax.plot(geom.edge_segments[e, p, :, 0], geom.edge_segments[e, p, :, 1], geom.edge_segments[e, p, :, 2], color='black')

    ax.scatter(geom.centroids[2][fstart:fend,0], geom.centroids[2][fstart:fend,1], geom.centroids[2][fstart:fend,2], s=100, marker="D", color='purple')

    foff = topo.kcells_off[2]
    eoff = topo.kcells_off[1]
    for f in range(fstart,fend):
        for eind,e in enumerate(topo.EF[f, :topo.nEF[f]]):
            for p in range(geom.num_edge_segments[e-eoff]):
                facetriangle = geom.face_triangles[f, eind, p, :, :]
                facenormal = geom.face_normals[f, eind, p, :]
                #print(f,eind,p,facetriangle[2,:],facenormal)
                ax.arrow3D(facetriangle[2,0], facetriangle[2,1], facetriangle[2,2], facenormal[0]*dx/4., facenormal[1]*dx/4., facenormal[2]*dx/4., mutation_scale=20, fc='green')


# cell centroids
# cell circumcenters
# cell quadrature

    ax = fig.add_subplot(2, 2, 3, projection='3d')

    ax.set_xlim3d(minx - xtot*0.1, maxx + xtot*0.1 + 1.3*dx)
    ax.set_ylim3d(miny - ytot*0.1, maxy + ytot*0.1 + 1.3*dx)
    ax.set_zlim3d(minz - ztot*0.1, maxz + ztot*0.1 + 1.3*dx)

    for e in range(estart,eend):
        for p in range(geom.num_edge_segments[e]):
            ax.plot(geom.edge_segments[e, p, :, 0], geom.edge_segments[e, p, :, 1], geom.edge_segments[e, p, :, 2], color='black')

    ax.scatter(geom.centroids[3][cstart:cend,0], geom.centroids[3][cstart:cend,1], geom.centroids[3][cstart:cend,2], s=100, marker="D", color='green')
    if topo.is_simplicial:
        ax.scatter(geom.circumcenters[cstart:cend,0], geom.circumcenters[cstart:cend,1], geom.circumcenters[cstart:cend,2], s=100, marker="D", color='blue')

    ax = fig.add_subplot(2, 2, 4, projection='3d')

    ax.set_xlim3d(minx - xtot*0.1, maxx + xtot*0.1 + 1.3*dx)
    ax.set_ylim3d(miny - ytot*0.1, maxy + ytot*0.1 + 1.3*dx)
    ax.set_zlim3d(minz - ztot*0.1, maxz + ztot*0.1 + 1.3*dx)

    for e in range(estart,eend):
        for p in range(geom.num_edge_segments[e]):
            ax.plot(geom.edge_segments[e, p, :, 0], geom.edge_segments[e, p, :, 1], geom.edge_segments[e, p, :, 2], color='black')

    ax.scatter(quad.quadpts[2][fstart:fend, :, :, :, 0], quad.quadpts[2][fstart:fend, :, :, :, 1], quad.quadpts[2][fstart:fend, :, :, :, 2], s=100, marker="2", color='black')

    ax.scatter(quad.quadpts[3][cstart:cend, :, :, :, :, 0], quad.quadpts[3][cstart:cend, :, :, :, :, 1], quad.quadpts[3][cstart:cend, :, :, :, :, 2], s=100, marker="x", color='black')

    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.close('all')

def plot1Dmeshpair():
    pass

def plot2Dmeshpair():
    pass

def plot3Dmeshpair():
    pass
