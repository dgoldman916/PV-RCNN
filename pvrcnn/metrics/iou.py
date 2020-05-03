    
"""
Area of Intersection of Two Rotated Rectangles

https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241#45268241
Author: Ruud de Jong

Here is a solution that does not use any libraries outside of Python's standard library.

Determining the area of the intersection of two rectangles can be divided in two subproblems:

Finding the intersection polygon, if any;
Determine the area of the intersection polygon.
Both problems are relatively easy when you work with the vertices (corners) of the rectangles.
So first you have to determine these vertices.
Assuming the coordinate origin is in the center of the rectangle,
the vertices are, starting from the lower left in a counter-clockwise direction:

(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), and (-w/2, h/2).

Rotating this over the angle a, and translating them to the proper position of the rectangle's center,
these become:

(cx + (-w/2)cos(a) - (-h/2)sin(a), cy + (-w/2)sin(a) + (-h/2)cos(a))

, and similar for the other corner points.

A simple way to determine the intersection polygon is the following:
you start with one rectangle as the candidate intersection polygon.
Then you apply the process of sequential cutting, in short:

- you take each edges of the second rectangle in turn,

- and remove all parts from the candidate intersection polygon
  that are on the "outer" half plane defined by the edge (extended in both directions).

Doing this for all edges leaves the candidate intersection polygon 
with only the parts that are inside the second rectangle or on its boundary.

The area of the resulting polygon (defined by a series of vertices)
can be calculated from the coordinates of the vertices.
You sum the cross products of the vertices of each edge (again in counter-clockwise order),
and divide that by two. See e.g. www.mathopenref.com/coordpolygonarea.html
"""

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
