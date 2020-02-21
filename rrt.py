# Thomas Proctor
# Jacqueline Doolittle
# Cozmo Team 54

import cozmo
import math
import sys
import time

from cmap import *
from rrt_gui import *
from node import *
from cozmo.util import radians, distance_mm, Angle, degrees

MAX_NODES = 20000
cozmo_global_pos = (152.4, 254, 0)


def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object

    distance = get_dist(node0, node1)
    new_node = node1
    new_node.parent = node0
    if distance > limit:
        # return new node in correct direction
        #                  /node1
        #                 /
        #                /
        #               /
        #              /|
        #             / |
        #            /  |
        #     limit /   |
        #          /    |
        #         /     |
        #   node0 -------
        #
        vector_angle = np.arctan2(node0[1], node0[0])
        add_x = np.cos(vector_angle) * limit  # limit is the hypotenuse
        add_y = np.sin(vector_angle) * limit
        coord = (node0[0] + add_x, node0[1] + add_y)
        new_node = Node(coord, node0)

    return new_node
    ############################################################################


def node_generator(cmap):
    width, height = cmap.get_size()
    rand_width = np.random.random() * width
    rand_height = np.random.random() * height
    rand_node = Node((rand_width, rand_height), None)
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    # pass

    while not cmap.is_inbound(rand_node) or cmap.is_inside_obstacles(rand_node):
        rand_width = np.random.random() * cmap.width
        rand_height = np.random.random() * cmap.height
        rand_node = Node((rand_width, rand_height), None)
    ############################################################################

    return rand_node


def RRT(cmap, start):
    cmap.add_node(start)

    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node
        #
        rand_node = cmap.get_random_valid_node()
        list_of_nodes = CozMap.get_nodes(cmap)
        nearest_node = list_of_nodes[0]
        distance = get_dist(list_of_nodes[0], rand_node)
        # Search through the list of nodes to find nearest
        for node in list_of_nodes:
            if get_dist(node, rand_node) < distance:
                nearest_node = node
                distance = get_dist(node, rand_node)
        # Create line segment from random to closest
        new_node = step_from_to(nearest_node, rand_node)
        while cmap.is_collision_with_obstacles((nearest_node, new_node)):
            # if we had to create a new node instead of use node1, the distance will initially be limit
            new_node = step_from_to(nearest_node, new_node, get_dist(nearest_node, new_node) - 0.5)
        pass
        ########################################################################
        time.sleep(0.01)
        cmap.add_path(nearest_node, new_node)
        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")


async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent

    ########################################################################
    # TODO: please enter your code below.
    # Set start node
    # cmap.set_start(start_node)
    # Start RRT
    RRT(cmap, cmap.get_start())
    path = cmap.get_smooth_path()
    index = 1
    # Path Planning for RRT
    while index < len(path):
        curr_node = path[index]
        dx = curr_node.x - path[index - 1].x
        dy = curr_node.y - path[index - 1].y
        # turn_angle = diff_heading_deg(cozmo.util, np.arctan2(dy, dx))
        # print("dx=", dx)
        # print("dy=", dy)
        # print("tan=", np.arctan2(dy, dx))
        await robot.turn_in_place(radians(np.arctan2(dy, dx)), angle_tolerance=Angle(1)).wait_for_completed()
        await robot.drive_straight(distance_mm(np.sqrt((dx ** 2) + (dy ** 2))),
                                   speed=cozmo.util.speed_mmps(50)).wait_for_completed()
        index += 1
    Visualizer.update(cmap)


def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object

        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.
    global_x = (node.x * np.cos(local_angle)) + (node.y * -(np.sin(local_angle))) + local_origin.x
    global_y = (node.x * np.sin(local_angle)) + (node.y * np.cos(local_angle)) + local_origin.y

    #new_node = None
    return Node((global_x, global_y))


async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap
    global cozmo_global_pos

    # Padding of objects and the robot for C-Space
    cube_padding = 60.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # The goal cube is defined as robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id
        if robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id:

            # Calculate the approach position of the object
            local_goal_pos = Node((0, -cozmo_padding))
            goal_pos = get_global_node(object_angle, object_pos, local_goal_pos)
            print(goal_pos.x)
            print(goal_pos.y)
            # Check whether this goal location is valid
            if cmap.is_inside_obstacles(goal_pos) or (not cmap.is_inbound(goal_pos)):
                print("The goal position is not valid. Please remove the goal cube and place in another position.")
            else:
                cmap.clear_goals()
                cmap.add_goal(goal_pos)
                goal_center = object_pos

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center
