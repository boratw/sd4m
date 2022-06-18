import datetime
import random
import os

def CreateLogPrefix():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')

def CreateLogFile(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    f = open(filename, "w")
    return f

def Get_Random_Color():
    c = [0, 0, 0]
    i = list(range(3))
    random.shuffle(i)
    c[i[0]] = random.randrange(128, 255)
    c[i[1]] = random.randrange(0, 255)
    c[i[2]] = random.randrange(0, 127)
    return c