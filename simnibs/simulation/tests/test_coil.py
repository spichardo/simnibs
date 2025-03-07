import os
import numpy as np
import nibabel as nib
import pytest

from ... import SIMNIBSDIR
from ...mesh_tools import mesh_io
from .. import coil_numpy as coil

@pytest.fixture(scope='module')
def sphere3_msh():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'sphere3.msh')
    return mesh_io.read_msh(fn)

@pytest.fixture(scope='module')
def testcoil_ccd():
    fn = os.path.join(
        SIMNIBSDIR, '_internal_resources', 'testing_files', 'testcoil.ccd')
    return fn

class TestReadCCD:
    def test_read_ccd(self):
        with open('test.ccd', 'w') as f:
            f.write('# number of elements\n')
            f.write('3\n')
            f.write('1e-003 3e-003 -4e-003 0e+000 0e+000 -4e-006\n')
            f.write('3e-003 1e-003  1e-003 1e+000 0e+000 0e-000\n')
            f.write('4e-003 2e-003 -5e-003 0e+000 1e+000 0e-000\n')
        position, dipole = coil.read_ccd('test.ccd')
        assert np.allclose(position, np.array([[1e-3, 3e-3, -4e-3],
                                               [3e-3, 1e-3, 1e-3],
                                               [4e-3, 2e-3, -5e-3]]))
        assert np.allclose(dipole, np.array([[0, 0, -4e-6],
                                             [1, 0, 0],
                                             [0, 1, 0]], dtype=float))
        os.remove('test.ccd')

    def test_parseccd(self, testcoil_ccd):
        d_position, d_moment, bb, res, info = coil.parseccd(testcoil_ccd)
        print(d_position)
        print(d_moment)
        print(bb)
        print(res)
        assert np.allclose(bb,((-100,100),(-100,100),(-100,100)))
        assert np.allclose(res,(10,10,10))
        assert info['coilname']=='Test coil'
        assert float(info['dIdtmax'])==100.0
        assert float(info['dIdtstim'])==100.0
        assert info['brand']=='None'
        assert info['stimulator']=='None'
        assert np.allclose(d_position, ((-1e-2,0,-1e-3),
                                        (1e-2,0,-1e-3),
                                        (-2e-2,2e-2,-2e-3),
                                        (2e-2,-2e-2,-2e-3)))
        assert np.allclose(d_moment, ((1e-6,0,2e-6),
                                      (-1e-6,0,-2e-6),
                                      (0,1e-6,0),
                                      (0,-1e-6,0)))

class TestCalcdAdt:
    def test_calc_dAdt_ccd(self):
        with open('test.ccd', 'w') as f:
            f.write('# number of elements\n')
            f.write('2\n')
            f.write('0e-000 0e-000  0e-000 0e+000 0e+000 1e-000\n')
            f.write('0e-000 0e-000  0e-000 0e+000 1e+000 0e-000\n')
        msh = mesh_io.Msh()
        # milimeters
        msh.nodes = mesh_io.Nodes(np.array([[1e3, 0., 0.], [0., 1e3, 0.], [0, 0., 1e3]]))
        coil_matrix = np.eye(4)
        dadt = coil._calculate_dadt_ccd(msh, 'test.ccd', coil_matrix, 1e6, None)
        assert np.allclose(dadt[[1, 2, 3]], np.array([[0., 0.1, -0.1],
                                                      [-0.1, 0, 0],
                                                      [0.1, 0., 0.]]))
        os.remove('test.ccd')

    def test_calc_dAdt_nifti(self, sphere3_msh):
        affine = np.array([[5., 0., 0., -300],
                           [0., 5., 0., -200],
                           [0., 0., 5., 0.],
                           [0., 0., 0., 1]])
        field = np.ones((121, 81, 41, 3))
        field[..., 1] = 2
        field[..., 2] = 3
        img = nib.Nifti1Image(field, affine)
        coil_matrix = np.array([[0., 1., 0., 0],
                                [1., 0., 0., 0],
                                [0., 0., 1., -100.],
                                [0., 0., 0., 1]])
        dadt = coil._calculate_dadt_nifti(sphere3_msh, img, coil_matrix, 1e6, None)
        sphere3_msh.nodedata.append(dadt)
        assert np.allclose(dadt.value[:, 0], 2e6, atol=1e-6)
        assert np.allclose(dadt.value[:, 1], 1e6, atol=1e-6)
        assert np.allclose(dadt.value[:, 2], 3e6, atol=1e-6)

    def test_calc_dAdt_ccd_fmm(self):
        try:
            import fmm3dpy
        except ImportError:
            print('skiping test, no fmm3dpy')
            return

        with open('test.ccd', 'w') as f:
            f.write('# number of elements\n')
            f.write('2\n')
            f.write('0e-000 0e-000  0e-000 0e+000 0e+000 1e-000\n')
            f.write('0e-000 0e-000  0e-000 0e+000 1e+000 0e-000\n')
        msh = mesh_io.Msh()
        # milimeters
        msh.nodes = mesh_io.Nodes(np.array([
            [1e3, 0., 0.],
            [0., 1e3, 0.],
            [0, 0., 1e3],
            [0.,0., -1e3]
        ]))
        coil_matrix = np.eye(4)
        dadt = coil._calculate_dadt_ccd_FMM(msh, 'test.ccd', coil_matrix, 1e6, None)
        assert np.allclose(
            dadt[:],
            np.array([
                [0., 0.1, -0.1],
                [-0.1, 0, 0],
                [0.1, 0., 0.],
                [-0.1, 0., 0.]
            ])
        )
        os.remove('test.ccd')
