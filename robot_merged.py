from controller import Robot, Motor, DistanceSensor, Camera
from collections import Counter


# ================= CONSTANTS =================

MAX_SPEED       = 6.28
MULTIPLIER      = 0.5           # General forward speed (50% max)

SENSOR_MAX      = 4096.0        # e-puck sensor max reading
OBSTACLE_THRESHOLD = 80.0       # Sensor value that triggers Braitenberg steering
STUCK_THRESHOLD    = 200.0      # All-front blockage → reverse

# Dog-specific RGB target (from capture.py)
DOG_R, DOG_G, DOG_B = 77, 80, 89
DOG_TOLERANCE = 40              # Per-channel tolerance for dog match

# General color detection threshold (from iseeRGB.py)
COLOR_THRESHOLD = 50
COLOR_DIFF      = 30

# Braitenberg steering weights [ps0 … ps7]
LEFT_WEIGHTS  = [ 0.9,  0.5,  0.2, -0.1, -0.1,  0.2,  0.5,  0.9]
RIGHT_WEIGHTS = [-0.9, -0.5, -0.2,  0.1,  0.1, -0.2, -0.5, -0.9]


# ================= DISTANCE SENSORS =================

def init_sensors(robot, timestep):
    """Initialise and return the 8 proximity sensor objects."""
    sensors = []
    for i in range(8):
        s = robot.getDevice(f"ps{i}")
        s.enable(timestep)
        sensors.append(s)
    return sensors


def read_sensors(sensors):
    """Return raw readings for all 8 sensors."""
    return [s.getValue() for s in sensors]


# ================= MOVEMENT =================

def set_velocity(left_motor, right_motor, left, right):
    """Clamp and apply wheel velocities."""
    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))


def move_forward(left_motor, right_motor):
    set_velocity(left_motor, right_motor,
                 MAX_SPEED * MULTIPLIER, MAX_SPEED * MULTIPLIER)


def stop(left_motor, right_motor):
    set_velocity(left_motor, right_motor, 0, 0)


def back_up(left_motor, right_motor):
    """Reverse with a slight curve to break out of corners."""
    set_velocity(left_motor, right_motor,
                 -0.3 * MAX_SPEED, -0.15 * MAX_SPEED)


# ================= OBSTACLE AVOIDANCE =================

def handle_obstacles(left_motor, right_motor, readings):
    """
    Braitenberg-based obstacle avoidance (from iseeRGB.py).
    Returns True while avoidance is active.
    """
    front = [readings[0], readings[1], readings[6], readings[7]]

    # Completely blocked → back up
    if all(v > STUCK_THRESHOLD for v in front):
        back_up(left_motor, right_motor)
        return True

    # Any sensor above threshold → steer away
    if any(v > OBSTACLE_THRESHOLD for v in readings):
        norm        = [v / SENSOR_MAX for v in readings]
        left_speed  = 0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED
        for i, n in enumerate(norm):
            left_speed  += LEFT_WEIGHTS[i]  * n * MAX_SPEED
            right_speed += RIGHT_WEIGHTS[i] * n * MAX_SPEED
        set_velocity(left_motor, right_motor, left_speed, right_speed)
        return True

    return False


# ================= CAMERA =================

def sample_region(cam, img, cx, cy, size=5):
    """
    Return the average RGB of a (2*size+1)² region centred on (cx, cy).
    Combines the robust averaging from iseeRGB.py with capture.py's camera use.
    """
    W = cam.getWidth()
    H = cam.getHeight()
    rs, gs, bs = [], [], []
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < W and 0 <= y < H:
                rs.append(cam.imageGetRed(img,  W, x, y))
                gs.append(cam.imageGetGreen(img, W, x, y))
                bs.append(cam.imageGetBlue(img,  W, x, y))
    return sum(rs) // len(rs), sum(gs) // len(gs), sum(bs) // len(bs)


# ================= DOG DETECTION =================

def is_dog(r, g, b):
    """
    True when the sampled colour is within DOG_TOLERANCE of the dog's
    known RGB signature (from capture.py).
    """
    return (abs(r - DOG_R) < DOG_TOLERANCE and
            abs(g - DOG_G) < DOG_TOLERANCE and
            abs(b - DOG_B) < DOG_TOLERANCE)


# ================= GENERAL COLOR DETECTION =================

def get_color_name(r, g, b):
    """
    Classify sampled RGB as 'red', 'green', 'blue', or 'NONE'
    (from iseeRGB.py).
    """
    if r > g + COLOR_DIFF and r > b + COLOR_DIFF and r > COLOR_THRESHOLD:
        return "red"
    if g > r + COLOR_DIFF and g > b + COLOR_DIFF and g > COLOR_THRESHOLD:
        return "green"
    if b > r + COLOR_DIFF and b > g + COLOR_DIFF and b > COLOR_THRESHOLD:
        return "blue"
    return "NONE"


# ================= IMAGE CAPTURE =================

def capture_image(camera, image_id):
    """Save a PNG snapshot from the camera (from capture.py)."""
    filename = f"dog_capture_{image_id}.png"
    camera.saveImage(filename, 100)
    print(f"[CAPTURE] Image saved: {filename}")


# ================= ENCOUNTER LOGGING =================

def on_color_event(color, encounter_log):
    """
    Log a newly detected general colour and print a running summary
    (from iseeRGB.py).
    """
    if color != "NONE":
        encounter_log.append(color)
        print(f"[COLOR] I see {color}")

    print("--- Color summary so far ---")
    if not encounter_log:
        print("  No colors detected yet.")
    else:
        counts = Counter(encounter_log)
        for c, n in counts.items():
            print(f"  {c}: {n} time(s)")
        print(f"  Order: {' -> '.join(encounter_log)}")
    print("----------------------------")


# ================= MAIN ROBOT =================

def run_robot(robot):

    timestep = int(robot.getBasicTimeStep())

    # --- Sensors ---
    sensors = init_sensors(robot, timestep)

    # --- Camera ---
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    cam_w = camera.getWidth()
    cam_h = camera.getHeight()

    # --- Motors ---
    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    stop(left_motor, right_motor)

    # --- State ---
    image_counter = 0       # Number of dog images saved
    dog_captured  = False   # Prevent repeated captures of the same encounter
    last_color    = None    # Last detected general colour
    encounter_log = []      # History of general colour encounters


    # ================= MAIN LOOP =================

    while robot.step(timestep) != -1:

        readings = read_sensors(sensors)
        img      = camera.getImage()
        r, g, b  = sample_region(camera, img, cam_w // 2, cam_h // 2)

        print(f"[RGB] r={r}  g={g}  b={b}")


        # -------- Dog detection (highest priority) --------

        if is_dog(r, g, b):
            print("[DOG] Dog detected — stopping.")
            stop(left_motor, right_motor)

            if not dog_captured:
                capture_image(camera, image_counter)
                image_counter += 1
                dog_captured = True

            # Skip obstacle avoidance and general colour logic while
            # the dog is in frame.
            continue

        # Reset capture flag once the dog leaves the frame
        dog_captured = False


        # -------- General colour logging --------

        detected_color = get_color_name(r, g, b)
        if detected_color != last_color:
            on_color_event(detected_color, encounter_log)
            last_color = detected_color


        # -------- Navigation --------

        avoiding = handle_obstacles(left_motor, right_motor, readings)
        if not avoiding:
            move_forward(left_motor, right_motor)


# ================= ENTRY POINT =================

if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)
