## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet
# Thomas Proctor
# Jacqueline Doolittle

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import cozmo
import imgclassifier
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image
from cozmo.util import degrees, distance_mm, speed_mmps

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from node import *
from cmap import *
from rrt import *


# init = True
#classifier = None
cozmo_location = (0, 0, 0)

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6, 10, 0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)
# init = True

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera. 
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)
    
    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)
    print(diag)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]
    # marker_list1 = []
    # for marker in markers:
    #     marker_list1.append((marker['xyh'], marker['label']))
    # # marker_list = [(marker['xyh'], marker['label']) for marker in markers]
    # marker_list = []
    # for marker in marker_list1:
    #     xyh = marker[0]
    #     label = marker[1]
    #     marker_list.append((xyh[0]/grid.scale, xyh[1]/grid.scale, xyh[2], label[0]))
    # marker_list = [(x/grid.scale, y/grid.scale, h, label) for x,y,h,label in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf, cozmo_location
    global cmap, rrt

    # chose to do 2 inches off the wall be corner locations
    corner_A = (2, (grid.height * 0.0393701) - 2)
    corner_B = ((grid.width * 0.0393701) - 2, (grid.height * 0.0393701) -2)
    corner_C = (2, 2)
    corner_D = ((grid.width * 0.0393701) - 2, 2)
    corners = [("A", corner_A), ("B", corner_B), ("C", corner_C), ("D", corner_D)]

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    # train cozmo so he can identify markers, but we only want to train him once -- done in markers/detect.py

    ###################

    # YOUR CODE HERE

    ###################
    """""""""
    Student answer is on the right track, no need to mess with threads yourself.A good non-blocking movement function 
    you can use is robot.drive_wheels.Once called the robot will continue moving until you call stop_all_motors.In the 
    meantime you can still perform particle updates until convergence.
    """""
    curr_pose = robot.pose


    # Obtain odometry information
    odometry = compute_odometry(curr_pose, cvt_inch=True)
    last_pose = robot.pose

    # Obtain list of currently seen markers and their poses
    robot_markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)
    # robot_markers = []
    # for marker in markers:
    #     robot_markers.append((marker[0], marker[1], marker[2]))
    # robot_markers = (robot_markers[0], robot_markers[1], robot_markers[2])
    # update the particle filter using the above information
    best_x, best_y, best_h, confidence = pf.update(odometry, robot_markers)
    print(best_x, best_y, best_h)
    # update the particle filter GUI
    # CAN GET RID OF THIS STUFF - WE KNOW THIS WORKS SO DON'T REALLY NEED THE GUI
    gui.show_mean(best_x, best_y, best_h, confidence)
    gui.show_particles(pf.particles)
    # gui.show_camera_image(camera_image)

    gui.updated.set()
    print("Confidence tested")
    if confidence:
        print("Received Confidence")
        # want Cozmo to stop looking
        robot.stop_all_motors()
        # we are now localized
        # Lab 2 code for scanning pictures & getting their locations (in the global scope) -- in markers/detect.py
        # can't scan as the same time as localizing because the global pos won't be correct yet

        # need to update this whenever we move, so we can keep track of Cozmo's global location
        cozmo_location = (best_x, best_y, best_h)

        # marker locations in inches
        identified_markers = dict()
        total_deg = 0
        # cozmo_angle = best_h
        marked = dict()
        cube_goals = []  # should hold all the goals because at first cozmo should be able to see the goals
        while total_deg < 360:
            # cozmo_angle = (cozmo_angle + 30) % 360
            await robot.turn_in_place(cozmo.util.degrees(30)).wait_for_completed()
            markers = await marker_processing(robot, camera_settings, show_diagnostic_image=False)
            for marker in markers:
                label = marker[3]
                if label not in identified_markers:
                    identified_markers[label] = (marker[0] * 0.0393701, marker[1] * 0.0393701, marker[2] * 0.0393701)
            updated, goal = await rrt.detect_cube_and_update_cmap(robot, marked, cozmo_location)
            if goal is not None and goal not in cube_goals:
                cube_goals.append(goal)
            total_deg += 30

        # Add obstacle to center
        obstacle_nodes = []
        cube_padding = 50.
        obstacle_nodes.append(
            get_global_node(0, Node((cmap.width / 2, cmap.height / 2)), Node((cube_padding, cube_padding))))
        obstacle_nodes.append(
            get_global_node(0, Node((cmap.width / 2, cmap.height / 2)), Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(
            get_global_node(0, Node((cmap.width / 2, cmap.height / 2)), Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(
            get_global_node(0, Node((cmap.width / 2, cmap.height / 2)), Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        # at this point he should be facing the same angle as he was before
        while len(cube_goals) > 0:
            # I think tom's RRT code chooses the goal but if not
            cmap.add_goal(cube_goals[0])
            cmap.reset()
            cmap.set_start(cozmo_location)
            # run RRT
            # rrt.RRT(cmap, cmap.get_start())  # maybe separate path planning method
            CozmoPlanning(robot)
            cozmo_location = cube_goals[0]
            # pick up cube (Lab 2 drone code)
            lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            cubes = robot.world.wait_until_observe_num_objects(num=1, object_type=cozmo.objects.LightCube, timeout=30)
            lookaround.stop()

            if len(cubes) == 0:
                robot.say_text("I don't see a cube").wait_for_completed()
            else:
                # maybe need to modify this to pick up the closest cube -- can test it out in practice
                robot.pickup_object(cubes[0], num_retries=10).wait_for_completed()
                # check which corner it was in (A B C D)
                min_dist = grid_distance(cozmo_location[0], corners[0][0], cozmo_location[1], corners[0][1])
                corner = "A"
                for i in range (0, len(corners)):
                    curr_dist = grid_distance(cozmo_location[0], corners[i][1][0], cozmo_location[1], corners[i][1][1])
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        corner = corners[i][0]
                goal = identified_markers["drone"]
                if corner is "A":
                    goal = identified_markers["drone"]
                elif corner is "B":
                    goal = identified_markers["plane"]
                elif corner is "C":
                    goal = identified_markers["inspection"]
                elif corner is "D":
                    goal = identified_markers["place"]
                cmap.reset()
                cmap.add_goal(goal)
                cmap.set_start(cozmo_location)  # not exact location because may move a little to pick up the cube
                # rrt.RRT(cmap, cmap.get_start())  # and do any path planning
                CozmoPlanning(robot)
                # manually set goal & run RRT
                # put down cube
                robot.place_object_on_ground_here(obj=cozmo.objects.LightCube).wait_for_completed()
                cozmo_location = goal

            # delete the cube from the list of goals & run RRT again -- repeat until all list of goals is empty
            cube_goals.pop(0)



        # picked_up = False
        # x, y, h = goal
        # p_x, p_y, p_h, p_confident = compute_mean_pose(pf.particles)
        # turn_angle = math.atan2(y - p_y, x - p_x)
        # turn_angle = diff_heading_deg(math.degrees(turn_angle), p_h)
        # await robot.turn_in_place(degrees(turn_angle)).wait_for_completed()
        # found_goal = False
        # distance_moved = 0
        # total_distance = math.sqrt((y - p_y)**2 + (x - p_x)**2) * 25.4
        # print(total_distance)
        # while not picked_up:
        #     # if not found_goal:
        #     #     await robot.drive_wheels(20, 20)
        #     # tolerance = 2
        #     print("x: " + str(p_x))
        #     print("y: " + str(p_y))
        #     # if (x - tolerance < r_x < x + tolerance and
        #     #       y - tolerance < r_y < y + tolerance):
        #     if distance_moved < total_distance:
        #         await robot.drive_straight(distance_mm(min(50, total_distance - distance_moved)), speed_mmps(50))\
        #             .wait_for_completed()
        #         distance_moved += 50
        #     else:
        #         found_goal = True
        #     if found_goal:
        #         robot.stop_all_motors()
        #         await robot.say_text("I win; I'm so handsome", play_excited_animation=True).wait_for_completed()
        #         found_goal = False
        #     if robot.is_picked_up:
        #         picked_up = True
        #         await robot.say_text("Put me down", play_excited_animation=False, in_parallel=True).wait_for_completed()
        #         pf.particles = Particle.create_random(PARTICLE_COUNT, grid)
        # await run(robot)
    else:
        # maybe have Cozmo look around
        #      init = False
        print("$$$")
        await robot.drive_wheels(10, 0)
        print("%%%")
        await run(robot)
        print("@@@")


class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()

