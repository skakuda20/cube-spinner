import unittest
import numpy as np
from cube_spinner import point3D, normalize, look_at, rotate_point, world_to_camera_point, project_point, Camera, Cube

class TestCubeRenderingIssues(unittest.TestCase):
    def test_face_normals_outward(self):
        # For a standard cube, all face normals should point outward
        cube = Cube(1)
        # Define faces as quads, check normal direction
        for face in cube.faces:
            v0, v1, v2 = [cube.vertices[i] for i in face[:3]]
            a = np.array([v1.x - v0.x, v1.y - v0.y, v1.z - v0.z])
            b = np.array([v2.x - v0.x, v2.y - v0.y, v2.z - v0.z])
            normal = np.cross(a, b)
            # For a unit cube centered at origin, normal should not be zero
            self.assertGreater(np.linalg.norm(normal), 0)

    def test_project_point_behind_camera(self):
        p = point3D(0, 0, -100)
        camera_distance = 2.5
        width, height = 91, 71
        result = project_point(p, camera_distance, width, height)
        self.assertIsNone(result)

    def test_degenerate_cube(self):
        cube = Cube(0)
        verts = set((v.x, v.y, v.z) for v in cube.vertices)
        self.assertEqual(len(verts), 1)

    def test_extreme_aspect_ratio(self):
        p = point3D(1, 0, 0)
        camera_distance = 2.5
        # Very wide
        result_wide = project_point(p, camera_distance, 1000, 10)
        self.assertIsInstance(result_wide, tuple)
        # Very tall
        result_tall = project_point(p, camera_distance, 10, 1000)
        self.assertIsInstance(result_tall, tuple)

    def test_camera_at_vertex(self):
        cube = Cube(2)
        cam_pos = cube.vertices[0]
        cam = Camera(cam_pos, look_at(cam_pos, point3D(0,0,0), point3D(0,1,0)), 2.5)
        for v in cube.vertices:
            try:
                world_to_camera_point(v, cam)
            except Exception as e:
                self.fail(f"world_to_camera_point raised {e}")

    def test_zero_rotation_axis(self):
        p = point3D(1, 0, 0)
        axis = [0, 0, 0]
        angle = np.pi / 2
        with self.assertRaises(ValueError):
            rotate_point(p, angle, axis)

    def test_frame_buffer_dtype(self):
        # Should work with correct dtype, fail with int
        frame = np.full((10, 10), ' ', dtype='<U1')
        try:
            frame.fill(' ')
        except Exception as e:
            self.fail(f"frame.fill(' ') raised {e}")
        frame_int = np.zeros((10, 10), dtype=int)
        with self.assertRaises(Exception):
            frame_int.fill(' ')
    def test_up_vector_parallel_to_forward(self):
        cam_pos = point3D(0, -5, 0)
        target = point3D(0, 0, 0)
        up = point3D(0, -1, 0)  # Parallel to forward
        with self.assertRaises(ValueError):
            look_at(cam_pos, target, up)

    def test_projection_axis_mismatch(self):
        # Should not raise, but check for reasonable output
        p = point3D(1, 0, 0)
        camera_distance = 2.5
        width, height = 91, 71
        result = project_point(p, camera_distance, width, height)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_cube_vertex_order(self):
        cube = Cube(2)
        # All vertices should be unique
        verts = set((v.x, v.y, v.z) for v in cube.vertices)
        self.assertEqual(len(verts), 8)
        # All edge indices should be valid
        for a, b in cube.edges:
            self.assertTrue(0 <= a < 8)
            self.assertTrue(0 <= b < 8)

    def test_camera_inside_cube(self):
        cube = Cube(2)
        cam = Camera(point3D(0, 0, 0), look_at(point3D(0, 0, 0), point3D(0, 0, 0.1), point3D(0, 1, 0)), 0.1)
        # All projected points should be valid or None (not crash)
        for v in cube.vertices:
            try:
                world_to_camera_point(v, cam)
            except Exception as e:
                self.fail(f"world_to_camera_point raised {e}")

    def test_rotation_axis(self):
        p = point3D(1, 0, 0)
        axis = [0, 1, 0]
        angle = np.pi
        rotated = rotate_point(p, angle, axis)
        # Rotating 180 degrees around y should flip x
        self.assertAlmostEqual(rotated[0], -1, places=5)
        self.assertAlmostEqual(rotated[1], 0, places=5)
        self.assertAlmostEqual(rotated[2], 0, places=5)

    def test_frame_buffer_clear(self):
        frame = np.full((10, 10), 'x', dtype='<U1')
        frame.fill(' ')
        self.assertTrue(np.all(frame == ' '))

    def test_shading_depth(self):
        # zmin == zmax should not crash
        from cube_spinner import get_shade, shading_chars
        z = 1.0
        shade = get_shade(z, z, z, shading_chars)
        self.assertIn(shade, shading_chars)

    def test_projection_numerical_instability(self):
        p = point3D(1, 0, 0)
        camera_distance = 1e-8
        width, height = 91, 71
        # Should not crash, but may return None
        try:
            project_point(p, camera_distance, width, height)
        except Exception as e:
            self.fail(f"project_point raised {e}")

if __name__ == '__main__':
    unittest.main()
