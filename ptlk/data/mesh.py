""" 3-d mesh reader """
import os
import copy
import numpy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot


class Mesh:
    def __init__(self):
        self._vertices = [] # array-like (N, D)
        self._faces = [] # array-like (M, K)
        self._edges = [] # array-like (L, 2)

    def clone(self):
        other = copy.deepcopy(self)
        return other

    def clear(self):
        for key in self.__dict__:
            self.__dict__[key] = []

    def add_attr(self, name):
        self.__dict__[name] = []

    @property
    def vertex_array(self):
        return numpy.array(self._vertices)

    @property
    def vertex_list(self):
        return list(map(tuple, self._vertices))

    @staticmethod
    def faces2polygons(faces, vertices):
        p = list(map(lambda face: \
                        list(map(lambda vidx: vertices[vidx], face)), faces))
        return p

    @property
    def polygon_list(self):
        p = Mesh.faces2polygons(self._faces, self._vertices)
        return p

    def plot(self, fig=None, ax=None, *args, **kwargs):
        p = self.polygon_list
        v = self.vertex_array
        if fig is None:
            fig = matplotlib.pyplot.gcf()
        if ax is None:
            ax = Axes3D(fig)
        if p:
            ax.add_collection3d(Poly3DCollection(p))
        if v.shape:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], *args, **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig, ax

    def on_unit_sphere(self, zero_mean=False):
        # radius == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = numpy.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        n = numpy.linalg.norm(v[:, 0:3], axis=1) # (N,)
        m = numpy.max(n) # scalar
        v[:, 0:3] = v[:, 0:3] / m
        self._vertices = v
        return self

    def on_unit_cube(self, zero_mean=False):
        # volume == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = numpy.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        m = numpy.max(numpy.abs(v)) # scalar
        v[:, 0:3] = v[:, 0:3] / (m * 2)
        self._vertices = v
        return self

    def rot_x(self):
        # camera local (up: +Y, front: -Z) -> model local (up: +Z, front: +Y).
        v = self.vertex_array
        t = numpy.copy(v[:, 1])
        v[:, 1] = -numpy.copy(v[:, 2])
        v[:, 2] = t
        self._vertices = list(map(tuple, v))
        return self

    def rot_zc(self):
        # R = [0, -1;
        #      1,  0]
        v = self.vertex_array
        x = numpy.copy(v[:, 0])
        y = numpy.copy(v[:, 1])
        v[:, 0] = -y
        v[:, 1] = x
        self._vertices = list(map(tuple, v))
        return self

def offread(filepath, points_only=True):
    """ read Geomview OFF file. """
    with open(filepath, 'r') as fin:
        mesh, fixme = _load_off(fin, points_only)
    if fixme:
        _fix_modelnet_broken_off(filepath)
    return mesh

def _load_off(fin, points_only):
    """ read Geomview OFF file. """
    mesh = Mesh()

    fixme = False
    sig = fin.readline().strip()
    if sig == 'OFF':
        line = fin.readline().strip()
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
    elif sig[0:3] == 'OFF': # ...broken data in ModelNet (missing '\n')...
        line = sig[3:]
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
        fixme = True
    else:
        raise RuntimeError('unknown format')

    for v in range(num_verts):
        vp = tuple(float(s) for s in fin.readline().strip().split(' '))
        mesh._vertices.append(vp)

    if points_only:
        return mesh, fixme

    for f in range(num_faces):
        fc = tuple([int(s) for s in fin.readline().strip().split(' ')][1:])
        mesh._faces.append(fc)

    return mesh, fixme

def _fix_modelnet_broken_off(filepath):
    oldfile = '{}.orig'.format(filepath)
    os.rename(filepath, oldfile)
    with open(oldfile, 'r') as fin:
        with open(filepath, 'w') as fout:
            sig = fin.readline().strip()
            line = sig[3:]
            print('OFF', file=fout)
            print(line, file=fout)
            for line in fin:
                print(line.strip(), file=fout)


def objread(filepath, points_only=True):
    """Loads a Wavefront OBJ file. """
    _vertices = []
    _normals = []
    _texcoords = []
    _faces = []
    _mtl_name = None

    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = tuple(map(float, values[1:4]))
            _vertices.append(v)
        elif values[0] == 'vn':
            v = tuple(map(float, values[1:4]))
            _normals.append(v)
        elif values[0] == 'vt':
            _texcoords.append(tuple(map(float, values[1:3])))
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'mtllib':
            _mtl_name = values[1]
        elif values[0] == 'f':
            face_ = []
            texcoords_ = []
            norms_ = []
            for v in values[1:]:
                w = v.split('/')
                face_.append(int(w[0]) - 1)
                if len(w) >= 2 and len(w[1]) > 0:
                    texcoords_.append(int(w[1]) - 1)
                else:
                    texcoords_.append(-1)
                if len(w) >= 3 and len(w[2]) > 0:
                    norms_.append(int(w[2]) - 1)
                else:
                    norms_.append(-1)
            #_faces.append((face_, norms_, texcoords_, material))
            _faces.append(face_)

    mesh = Mesh()
    mesh._vertices = _vertices
    if points_only:
        return mesh

    mesh._faces = _faces

    return mesh


if __name__ == '__main__':
    def test1():
        mesh = objread('model_normalized.obj', points_only=False)
        #mesh.on_unit_sphere()
        mesh.rot_x()
        mesh.plot(c='m')
        matplotlib.pyplot.show()
    test1()

#EOF