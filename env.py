import rospy
import pickle
import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState


class Environment:

    def __init__(self, objects, rng_ranges=None):
        self.publisher = rospy.Publisher(
            "/gazebo/set_model_state",
            ModelState,
            queue_size=10)
        self.contact_listener = rospy.Subscriber(
            "/white_bumper",
            ContactsState,
            callback=self.is_contact,
            queue_size=10
        )
        self.objects = objects
        self.num_objects = len(objects)
        self.rng_ranges = rng_ranges
        self.contacts = [0] * (self.num_objects)
        self.prev_state = None

    def get_reward(self):
        # white = np.array(self.get_object_position("white_ball"))
        # red = np.array(self.get_object_position("red_ball"))
        red_contacts = self.contacts[1]
        yellow_contacts = self.contacts[2]
        if red_contacts > 0 and yellow_contacts > 0:
            reward = 10.0
        elif red_contacts > 0 or yellow_contacts > 0:
            reward = 0.0
        else:
            # reward = -np.linalg.norm(white-red, 2)
            reward = -10.0
        # reward = -np.linalg.norm(white-red, 2) + red_contacts * 1.0

        return reward

    def get_state(self):
        msg = self.get_model_states()
        indices = list(map(lambda x: msg.name.index(x), self.objects))
        poses = np.zeros((self.num_objects, 2))

        for i in range(self.num_objects):
            p = msg.pose[indices[i]]
            poses[i] = [p.position.x, p.position.y]
        return poses.reshape(-1)

    def save(self, filename):
        dic = {"name": [], "position": [], "orientation": []}
        msg = self.get_model_states()

        for i in range(len(msg.name)):
            dic["name"].append(msg.name[i])
            p = msg.pose[i].position
            o = msg.pose[i].orientation
            p = [p.x, p.y, p.z]
            o = [o.x, o.y, o.z, o.w]
            dic["position"].append(p)
            dic["orientation"].append(o)

        pickle.dump(dic, open(filename, "wb"))

    def load(self, filename):
        dic = pickle.load(open(filename, "rb"))

        for i in range(len(dic["name"])):
            self.set_model_state(
                dic["name"][i],
                dic["position"][i],
                dic["orientation"][i])

    def random(self):
        self.contacts = [0] * self.num_objects
        positions = self.get_state().reshape(-1, 2)
        current = []
        proposed = []
        for p in positions:
            current.append(p)

        for i in range(self.num_objects):
            valid = False
            while not valid:
                valid = True
                scale = self.rng_ranges[i][:3]
                offset = self.rng_ranges[i][3:]
                pos = np.random.rand(3) * scale + offset
                if i != 0:
                    for p in current:
                        if np.linalg.norm(p[:2] - pos[:2]) < 0.1:
                            valid = False
                            break
            # rospy.loginfo("{0} object is reset".format(i))
            current.append(pos)
            proposed.append(pos)

        self.prev_state = proposed
        for i, pos in enumerate(proposed):
            self.set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def load_prev_state(self):
        for i, pos in enumerate(self.prev_state):
            self.set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def is_stationary(self):
        msg = self.get_model_states()
        indices = list(map(lambda x: msg.name.index(x), self.objects))
        if self.contacts[1] > 0 and self.contacts[2] > 0:
            return True
        for i, obj in enumerate(msg.twist):
            if i not in indices:
                continue

            if abs(obj.linear.x) > 1e-2:
                return False

            if abs(obj.linear.y) > 1e-2:
                return False

        return True

    def get_model_states(self):
        msg = rospy.wait_for_message("/model_states", ModelStates)
        return msg

    def get_object_position(self, name):
        msg = self.get_model_states()
        idx = msg.name.index(name)
        position = msg.pose[idx].position
        return [position.x, position.y]

    def set_model_state(self, name, pos, quat):
        msg = ModelState()
        msg.model_name = name
        msg.pose = Pose()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.publisher.publish(msg)

    def is_contact(self, msg):
        for i in range(len(msg.states)):
            collision1 = msg.states[i].collision1_name
            collision2 = msg.states[i].collision2_name
            idx1 = collision1.index(":")
            idx2 = collision2.index(":")
            if collision1[:idx1] == "white_ball":
                idx = idx2
                collision = collision2
            else:
                idx = idx1
                collision = collision1

            if collision[:idx] == "small_table" or collision[:idx] == "torobo" or collision[:idx] == "ground_plane":
                continue
            else:
                self.contacts[self.objects.index(collision[:idx])] += 1
