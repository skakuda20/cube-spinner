import os
import time
import yaml
import numpy as np

shading_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']

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
    
    def cross(self, other):
        return point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x - other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __str__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

class Cube:
    def __init__(self, scale):
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

        for i, v in enumerate(self.vertices):
            self.vertices[i] = [coord * scale for coord in v]

        self.edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]

        self.faces = [
            [0, 1, 3, 2],  # front
            [4, 5, 7, 6],  # back
            [0, 1, 5, 4],  # bottom
            [2, 3, 7, 6],  # top
            [0, 2, 6, 4],  # left
            [1, 3, 7, 5],  # right
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


def normalize(vector):
    magnitude = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return point3D(vector.x / magnitude, vector.y / magnitude, vector.z / magnitude)


def look_at(camera_position, target_position, up_vector):
    forward = normalize(target_position - camera_position)
    right = normalize(up_vector.cross(forward))

    # Create 3x3 rotation matrix from right, up and forward vectors
    rotation_matrix = [
        [right.x, right.y, right.z],
        [up_vector.x, up_vector.y, up_vector.z],
        [forward.x, forward.y, forward.z]
    ]

    # Convert rotation matrix to quaternion
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion


def euclidian_distance(point1, point2):
    return np.sqrt((point2.x-point1.x)**2 + (point2.y-point1.y)**2 + (point2.z-point1.z)**2)


def rotation_matrix_to_quaternion(R):
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]
    
    tr = r00 + r11 + r22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (r21 - r12) / S
        y = (r02 - r20) / S
        z = (r10 - r01) / S
    elif r00 > r11 and r00 > r22:
        S = np.sqrt(1.0 + r00 - r11 - r22) * 2
        w = (r21 - r12) / S
        x = 0.25 * S
        y = (r01 + r10) / S
        z = (r02 + r20) / S
    elif r11 > r22:
        S = np.sqrt(1.0 + r11 - r00 - r22) * 2
        w = (r02 - r20) / S
        x = (r01 + r10) / S
        y = 0.25 * S
        z = (r12 + r21) / S
    else:
        S = np.sqrt(1.0 + r22 - r00 - r11) * 2
        w = (r10 - r01) / S
        x = (r02 + r20) / S
        y = (r12 + r21) / S
        z = 0.25 * S

    return Quaternion(w, x, y, z)

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

    z = point.z + camera_distance
    if z <= 0:
        return None

    x2d = point.x * factor
    y2d = point.y * factor

    # Convert to screen coordinates
    screen_x = int(width / 2 + x2d)
    screen_y = int(height / 2 - y2d)

    return (screen_x, screen_y)



def draw_line(frame, x0, y0, x1, y1, char):
    height, width = frame.shape

    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        if 0 <= int(x0) < width and 0 <= int(y0) < height:
            frame[int(y0), int(x0)] = char
        return

    x_inc = dx / steps
    y_inc = dy / steps

    x = x0
    y = y0
    for i in range(steps + 1):  # include endpoint
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            frame[yi, xi] = char
        x += x_inc
        y += y_inc


def draw_scene(points, edges, camera, frame):
    height, width = frame.shape
    frame.fill(0)

    projected_vertices = []

    for point in points:
        camera_point = world_to_camera_point(point3D(point[0], point[1], point[2]), camera)
        result = project_point(camera_point, camera.focal_len, width, height)
        projected_vertices.append(result)

    for edge in edges:
        start, end = edge

        p0 = projected_vertices[start]
        p1 = projected_vertices[end]

        if p0 is not None and p1 is not None:
            x0, y0 = p0
            x1, y1 = p1
            draw_line(frame, x0, y0, x1, y1, 1)

    print(frame)


def dev_testing():
    cube = Cube(2)

    # projection = Projection()
    height = 21
    width = 21

    frame = np.zeros((height, width), dtype=np.uint8)

    camera_distance = 2
    camera_position = point3D(2, 2, -camera_distance)
    target_position = point3D(0, 0, 0)  # The point to look at (cube center)
    up_vector = point3D(0, 1, 0)        # World up direction

    orientation = look_at(camera_position, target_position, up_vector)
    camera = Camera(camera_position, orientation, camera_distance)

    draw_scene(cube.vertices, cube.edges, camera, frame)


    # Test with rotation around z-axis
    axis = [0, 0, 1]
    angle = np.pi / 12
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        for i, vertex in enumerate(cube.vertices):
            rotated = rotate_point(vertex, angle, axis)
            # print(f"{vertex} --> {rotated}")
            cube.vertices[i] = rotated

        draw_scene(cube.vertices, cube.edges, camera, frame)
        time.sleep(0.1)


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
