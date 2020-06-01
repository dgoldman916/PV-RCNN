import numpy as np


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = np.pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*np.cos(angle)
    dxsin = dx*np.sin(angle)
    dycos = dy*np.cos(angle)
    dysin = dy*np.sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def get_intersection_area(r1, r2):
    """
    Area of Intersection of Two Rotated Rectangles

    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241#45268241
    Author: Ruud de Jong
    """

    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1
    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection
            
        # create edge
        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


def get_intersection_height(h1, h2):
    # array of centers
    centers = np.array([h1[0], h2[0]])
    # array of heights
    heights = np.array([h1[1], h2[1]])
    # calculate box tops
    tops = centers + heights / 2
    # calculate box bottoms
    bottoms = centers - heights / 2
    # set lower top
    intersection_top = min(tops)
    # the bottom intersection between the boxes will
    # always occur at the bottom of the higher box
    intersection_bottom = bottoms[np.argmax(centers)]
    # if the bottom intersection is higher than the
    # top intersection, they don't intersect.
    if intersection_bottom > intersection_top:
        return 0.0
    # return the difference between the top intersection
    return intersection_top - intersection_bottom    


def get_iou(bb1, bb2):
    # iou = intersection / union
    intersection = get_intersection_volume(bb1, bb2)
    union = get_union_volume(bb1, bb2)
    iou = intersection / union
    return iou
    

def get_intersection_volume(bb1, bb2):
    to_rect = lambda bb: (bb[0], bb[1], bb[3], bb[4], bb[6])
    to_height = lambda bb: (bb[2], bb[5])
    # calculate area of intersection between boxes (2d)
    i_area = get_intersection_area(to_rect(bb1), to_rect(bb2))
    # calculate height of intersection between box heights (1d)
    i_height = get_intersection_height(to_height(bb1), to_height(bb2))   
    # multiply area * height to get intersection volume
    intersection_volume = i_area * i_height
    return intersection_volume
    
    
def get_union_volume(bb1, bb2):
    # rectangle volume = w*h*l
    get_rectangle_volume = lambda w, h, l: w*h*l
    # union = volume of two boxes - intersection of boxes
    union_volume = get_rectangle_volume(*bb1[3:6]) + get_rectangle_volume(*bb2[3:6]) - get_intersection_volume(bb1, bb2)
    return union_volume


def test_single_case_bb_iou(bb1=None, bb2=None, expected_val=None):
    if expected_val:
        assert isinstance(expected_val, float)
    if not bb1:
        bb1 = [0, 0, 0, 1, 1, 1, np.pi/2]
    if not bb2:
        bb2 = [0.5, 0.5, 0.5, 1, 1, 1, 0]
    iou = get_iou(bb1, bb2)
    if expected_val:
        if round(iou, 2) == round(expected_val, 2):
            return True
        else:
            return False
    return iou



def test_bb_iou():
    # label_fpath = "../data/kitti/training/label_2/000037.txt"
    #image_fpath = '../data/kitti/training/velodyne_reduced/000032.bin'
    # boxes, batch_idx, class_idx, scores = inference.main(image_fpath)
    boxes = np.array([
            [19.5561,  0.5630, -0.5180,  1.5810,  3.5372,  1.4999,  1.6627],
            [ 0.1940,  7.6636, -0.9024,  1.5800,  3.6634,  1.4650,  1.7097],
            [31.3177,  3.9726, -0.3252,  1.6397,  4.1524,  1.5230,  1.5910]
        ])
    ###########################
    # dev
    ###########################
    # label_header = [
    #  "type", "truncated", "occluded", "alpha", "bbox1", "bbox2", "bbox3", "bbox4",
    #  "w", "h", "l", "x", "y", "z", "yaw"
    #]
    box_order = ("x", "y", "z", "w", "l", "h", "yaw")
    # labels = [dict(zip(label_header, label.strip().split(" "))) for label in open(label_fpath).readlines()]
    ###########################
    # get max iou between label and proposal
    gt_proposal_match_with_iou = {}
    for box_idx in range(boxes.shape[0]):
        label_proposal_match_with_iou[box_idx] = 0
        max_iou = 0
        for label_idx, label in enumerate(labels):
            # label_bbox = np.array([float(label[key]) for key in box_order])
            proposal_bbox = boxes[box_idx]
            iou = get_iou(proposal_bbox, label_bbox)
            print(iou)
            if iou > max_iou:
                max_iou = iou
                label_proposal_match_with_iou[box_idx] = label_idx
    #iou = bb_intersection_over_union(proposal_bbox, label_bbox)
    return label_proposal_match_with_iou
