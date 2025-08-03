import unittest
import numpy as np
from cube_spinner import point3D, normalize, look_at, rotate_point, world_to_camera_point, project_point, Camera, Cube

class TestCubeSpinner(unittest.TestCase):
    def test_normalize(self):
        v = point3D(3, 0, 4)
        n = normalize(v)
        self.assertAlmostEqual(np.sqrt(n.x**2 + n.y**2 + n.z**2), 1.0)

    def test_look_at_right_orientation(self):
        cam_pos = point3D(0, -5, 0)
        target = point3D(0, 0, 0)
        up = point3D(0, 0, 1)
        q = look_at(cam_pos, target, up)
        self.assertIsInstance(q, type(q))

    def test_rotate_point_identity(self):
        p = point3D(1, 2, 3)
        axis = [0, 1, 0]
        angle = 0
        rotated = rotate_point(p, angle, axis)
        self.assertTrue(np.allclose([p.x, p.y, p.z], rotated))

    def test_world_to_camera_point(self):
        cam = Camera(point3D(0, -5, 0), look_at(point3D(0, -5, 0), point3D(0, 0, 0), point3D(0, 0, 1)), 2.5)
        p = point3D(0, 0, 0)
        cam_p = world_to_camera_point(p, cam)
        self.assertIsInstance(cam_p, point3D)

    def test_project_point_in_front(self):
        p = point3D(1, 0, 0)
        camera_distance = 2.5
        width, height = 91, 71
        result = project_point(p, camera_distance, width, height)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_project_point_behind(self):
        p = point3D(1, 0, -10)
        camera_distance = 2.5
        width, height = 91, 71
        result = project_point(p, camera_distance, width, height)
        self.assertIsNone(result)

    def test_cube_vertices_unique(self):
        cube = Cube(2)
        verts = set((v.x, v.y, v.z) for v in cube.vertices)
        self.assertEqual(len(verts), 8)

if __name__ == '__main__':
    unittest.main()
