import xml.etree.ElementTree as elemTree

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetBody(goal, name) :
    tree = elemTree.parse("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/hopper.xml")
    for body in tree.iter("body"):
        if "name" in body.attrib:
            if(body.attrib["name"] == "torso"):
                geom = body.find("geom")
                geom.attrib["density"] = str(goal[0] * 1000)

    tree.write("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/" + name)
