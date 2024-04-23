#!/usr/bin/env python3

MAP_WIDTH = 100
MAP_HEIGHT = 100
TOWERS = 10
POINTS = 10000

import random
random.seed(12)

def generate():
    towers = []
    for i in range(TOWERS):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        towers.append((x,y))

    points = []
    for i in range(POINTS):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        points.append((x, y))

    return towers, points


def dist(point, tower):
    x1, y1 = point
    x2, y2 = tower
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def training_data(points, towers):
    rows = []
    for p in points:
        row = []
        x, y = p
        row.append(str(x))
        # row.append(str(y))

        for t in towers:
            row.append("{:.3f}".format(dist(p, t)))
        rows.append(','.join(row))
    return '\n'.join(rows)

def tower_positions(towers):
    row = []
    for (x, y) in towers:
       row.append(str((x,y)))
    return ';'.join(row)

# print(tower_positions(towers))
#
#



towers, points = generate()
training_points = points[:-100]
validation_points = points[-100:]
# print(training_data(points, towers))

for vp in validation_points:
    st = str(vp)
    for t in towers:
        st += "\t" + str("{:.3f}".format(dist(vp, t)))
    print(st)
