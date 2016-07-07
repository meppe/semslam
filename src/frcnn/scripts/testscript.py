#!/usr/bin/env python

import rospy
import rosgraph

master_uri = rosgraph.get_master_uri()
print("masteruri")
print(master_uri)
rospy.init_node("test")

print("done init node")