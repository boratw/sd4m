import xml.etree.ElementTree as elemTree

def modifystr(s, length):
    strs = s.split(" ")
    if len(strs) == 3:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length)
    elif len(strs) == 6:
        return str(float(strs[0]) * length) + " " + str(float(strs[1]) * length) + " " + str(float(strs[2]) * length) + " " + str(float(strs[3]) * length) + " " + str(float(strs[4]) * length) + " " + str(float(strs[5]) * length)

def SetLeg(goal, name) :
    tree = elemTree.parse("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/ant.xml")
    for body in tree.iter("body"):
        if "name" in body.attrib:
            if(body.attrib["name"] == "aux_1"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[0])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
            if(body.attrib["name"] == "aux_2"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[1])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
            if(body.attrib["name"] == "aux_3"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[2])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
            if(body.attrib["name"] == "aux_4"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[3])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])

    tree.write("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/" + name)

def SetLeg6(goal, name) :
    tree = elemTree.parse("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/ant6.xml")
    for body in tree.iter("body"):
        if "name" in body.attrib:
            if(body.attrib["name"] == "aux_1"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[0])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[0])
            if(body.attrib["name"] == "aux_2"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[1])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[1])
            if(body.attrib["name"] == "aux_3"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[2])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[2])
            if(body.attrib["name"] == "aux_4"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[3])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[3])
            if(body.attrib["name"] == "aux_5"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[4])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[4])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[4])
            if(body.attrib["name"] == "aux_6"):
                geom = body.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[5])
                body2 = body.find("body")
                body2.attrib["pos"] = modifystr(body2.attrib["pos"], goal[5])
                geom = body2.find("geom")
                geom.attrib["fromto"] = modifystr(geom.attrib["fromto"], goal[5])


    tree.write("/home/user/Documents/Taewoo/gym/gym/envs/mujoco/assets/" + name)

SetLeg([1., 1., 1., 1.], "ant_noleg0.xml")
SetLeg([0.001, 1., 1., 1.], "ant_noleg1.xml")
SetLeg([1., 0.001, 1., 1.], "ant_noleg2.xml")