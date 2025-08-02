import yaml
import numpy as np


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

class point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return point3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __str__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

class Cube:
    def __init__(self):
        self.vertices = [
            [ 1, 1, 1],
            [ 1,-1, 1],
            [-1,-1, 1],
            [-1, 1, 1],
            [ 1, 1,-1],
            [ 1,-1,-1],
            [-1,-1,-1],
            [-1, 1,-1],
        ]

        self.center = [0, 0, 0]

        self.orientation = Quaternion(1, 0, 0, 0)

class Camera:
    def __init__(self, position, orientation, focal_len):
        self.position = position  # [x, y, z]
        self.orientation = orientation  # Quaternion
        self.focal_len = focal_len


class Projection:
    def __init__(self, camera, fov, width, height):
        self.camera = camera
        self.fov = fov  # Field of view in radians
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)  # Placeholder for the frame buffer

    def project(self, point):
        # Placeholder for projection logic
        pass


def euclidian_distance(point1, point2):
    return np.sqrt((point2.x-point1.x)**2 + (point2.y-point1.y)**2 + (point2.z-point1.z)**2)


def quaternion_multiply(q1, q2):
    # Extract components from quaternions
    w1 = q1.w
    x1 = q1.x
    y1 = q1.y
    z1 = q1.z

    w2 = q2.w
    x2 = q2.x
    y2 = q2.y
    z2 = q2.z

    # Quaternion multiplication formula (binomial expanssion)
    return Quaternion(
        w1*w2 - x1*x2 - y1*y2 - z1*z2, # Scalar part
        w1*x2 + x1*w2 + y1*z2 - z1*y2, # i component
        w1*y2 - x1*z2 + y1*w2 + z1*x2, # j component
        w1*z2 + x1*y2 - y1*x2 + z1*w2  # k component
        )


def quaternion_conjugate(q):
    return Quaternion(q.w, -q.x, -q.y, -q.z)


def rotate_point(point, angle, axis):
    q = make_rotation_quaternion(axis, angle)
    p = Quaternion(0, *point)
    q_conj = quaternion_conjugate(q)

    # Rotate point using quaternion multiplication
    rotated_point = quaternion_multiply(quaternion_multiply(q, p), q_conj)
    return [rotated_point.x, rotated_point.y, rotated_point.z]


def make_rotation_quaternion(axis, angle):
    x, y, z = axis
    s = np.sin(angle / 2)   # amount of imaginary part
    w = np.cos(angle / 2)   # scalar part

    return Quaternion(w, x*s, y*s, z*s)


def rotate_point_by_quaternion(point, q):
    p = Quaternion(0, point.x, point.y, point.z)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, p), q_conj)
    return point3D(rotated.x, rotated.y, rotated.z)


def world_to_camera_point(point, camera):
    relative_point = point - camera.position
    p = Quaternion(0, relative_point.x, relative_point.y, relative_point.z)

    # camera_space_point = rotate_point_by_quaternion(p, quaternion_conjugate(camera.orientation))
    camera_space_q = quaternion_multiply(p, quaternion_conjugate(camera.orientation))
    return point3D(camera_space_q.x, camera_space_q.y, camera_space_q.z)


def project_point(point, camera_distance, width, height, scale=6):
    # Apply prespective projection
    factor = scale / (point.z + camera_distance)
    x2d = point.x * factor
    y2d = point.y * factor

    # Convert to screen coordinates
    screen_x = int(width / 2 + x2d)
    screen_y = int(height / 2 - y2d)

    return (screen_x, screen_y)


def draw_scene(points, camera, frame):
    height, width = frame.shape
    frame.fill(0)
    for point in points:
        camera_point = world_to_camera_point(point3D(point[0], point[1], point[2]), camera)
        x, y = project_point(camera_point, camera.focal_len, width, height)
        frame[y, x] = 1
    
    print(frame)


def dev_testing():
    cube = Cube()

    # projection = Projection()
    height = 15
    width = 15

    frame = np.zeros((height, width), dtype=np.uint8)

    camera_distance = 2
    camera = Camera(point3D(0, 0, -camera_distance), Quaternion(1, 0, 0, 0), camera_distance)

    draw_scene(cube.vertices, camera, frame)


    # Test with rotation around z-axis
    axis = [0, 0, 1]
    angle = np.pi / 4  # 45 degrees
    
    for i, vertex in enumerate(cube.vertices):
        rotated = rotate_point(vertex, angle, axis)
        print(f"{vertex} --> {rotated}")
        cube.vertices[i] = rotated

    draw_scene(cube.vertices, camera, frame)


def main():
    angle = 0

    while True:
        # 1. Generate quaternion from current angle

        # 2. Rotate all cube vertices

        # 3. Project 3D points to 2D

        # 4. Draw the projected points

        # 5. Clear screen and wait

        pass


if __name__ == "__main__":
    # Parse args
    with open("config.yaml", "r") as file:
        config_data = yaml.safe_load(file)

    dev_testing()
    # main()
