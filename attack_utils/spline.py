from os import stat
import numpy as np

def ifcross(p1, p2, p, px, py):
    d1 = (p1[0]-px)*(p[1]-py) - (p1[1]-py)*(p[0]-px)
    d2 = (p2[0]-px)*(p[1]-py) - (p2[1]-py)*(p[0]-px)
    if d1 * d2 < 0:
        return False
    else:
        return True

def compute_dis(p, px, py):
    dis = pow(pow(p[0]-px, 2) + pow(p[1]-py, 2), 0.5)
    return dis

def cross(p1, p2, p3, p4):
    v1x = p1[0] - p2[0] # x1-x2
    v1y = p1[1] - p2[1] # y1-y2
    v2x = p3[0] - p4[0] # x3-x4
    v2y = p3[1] - p4[1] # y3-y4
    return v1x * v2y - v1y * v2x # (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    
def ifCross_before(poly, cur_point, index):
    """
    param poly:[[],[],...,[]]
    param cur_point:[x,y]
    index: 
    return True|False
    """
    for i, point in enumerate(poly):
        if i==(index + len(poly) - 1) % len(poly) or i == index:
            continue
        next_i = (i+1) % len(poly)
        before_index = (index - 1 + len(poly)) % len(poly)
        # pa, pb, pc is point_i, point_next_i, cur_point's before, cur_point
        pa, pb, pc, pd = point, poly[next_i], poly[before_index], cur_point
        if np.abs((pa[1] - pb[1]) * (pc[0] - pd[0])) == np.abs((pc[1] - pd[1]) * (pa[0] - pb[0])): # judge parallel
            continue
        if np.abs((pa[1] - pb[1]) * (pb[0] - pd[0])) == np.abs((pb[1] - pd[1]) * (pa[0] - pb[0])): # (y1-y2)*(x2-x4)==(y2-y4)*(x1-x2)
            if min(pa[1], pb[1]) <= pd[1] <= max(pa[1], pb[1]) and min(pa[0],pb[0]) <= pd[0] <= max(pa[0],pb[0]): # if min(y1,y2) <= y4 <= max(y1,y2)
                return True
        if cross(pa, pb, pb, pc) * cross(pa, pb, pb, pd) < 0 and cross(pc, pd, pd, pa) * cross(pc, pd, pd, pb) < 0: 
            return True
    return False

def ifCross_after(poly, cur_point, index):
    """
    param poly:[[],[],...,[]]
    param cur_point:[x,y]
    index: 
    return True|False
    """
    for i, point in enumerate(poly):
        if i == (index + len(poly) - 1) % len(poly) or i == index:
            continue
        next_i = (i+1) % len(poly)
        after_index = (index + 1) % len(poly)
        # pa, pb, pc is point_i, point_next_i, cur_point's after, cur_point
        pa, pb, pc, pd = point, poly[next_i], poly[after_index], cur_point
        if np.abs((pa[1] - pb[1]) * (pc[0] - pd[0])) == np.abs((pc[1] - pd[1]) * (pa[0] - pb[0])): # judge parallel
            continue
        if np.abs((pa[1] - pb[1]) * (pb[0] - pd[0])) == np.abs((pb[1] - pd[1]) * (pa[0] - pb[0])): # (y1-y2)*(x2-x4)==(y2-y4)*(x1-x2)
            if min(pa[1], pb[1]) <= pd[1] <= max(pa[1], pb[1]) and min(pa[0], pb[0]) <= pd[0] <= max(pa[0], pb[0]): # if min(y1,y2) <= y4 <= max(y1,y2)
                return True
        if cross(pa, pb, pb, pc) * cross(pa, pb, pb, pd) < 0 and cross(pc, pd, pd, pa) * cross(pc, pd, pd, pb) < 0: 
            return True
    return False

# End-to-end to ensure a closed curve
def CatmullRomSpline(alpha, P0, P1, P2, P3, npoints=100):
    """
    P0, P1, P2, and P3 should be (x,y,z) point triples that define the Catmull-Rom spline.
    npoints is the number of points to include in this curve segment.
    """
    # Convert the points to numpy so that we can do array multiplication
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])

    # Calculate t0 to t4
    #   alpha = 0.5
    def tj(ti, Pi, Pj):
        return sum([(i-j)**2 for i,j in zip(Pi,Pj)])**alpha + ti

    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    # Only calculate points between P1 and P2
    t = np.linspace(t1,t2,npoints)

    # Reshape so that we can multiply by the points P0 to P3
    # and get a point for each value of t.
    t = t.reshape(len(t),1)
    A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
    A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
    A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
    
    B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
    B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

    C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
    return C

def CatmullRomChain(alpha, P, nP=100):
    """
    Calculate Catmull Rom for a chain of points and return the combined curve.
    """
    sz = len(P)

    # The curve C will contain an array of (x,y,z) points.
    C = []
    for i in range(sz):
        i = i
        c = CatmullRomSpline(alpha, P[i], P[(i+1)%sz], P[(i+2)%sz], P[(i+3)%sz], nP)
        C.extend(c)
    return C

def if_cross(p1, p2, p):
    px = 49 
    py = 49
    d1 = (p1[0]-px)*(p[1]-py) - (p1[1]-py)*(p[0]-px)
    d2 = (p2[0]-px)*(p[1]-py) - (p2[1]-py)*(p[0]-px)
    print(d1,d2)
    if d1 * d2 < 0:
        return False
    else:
        return True

def spline_mask(points, H, W):
    # print(H,W)
    alpha = 0.5
    # Calculate the Catmull-Rom splines through the points
    c = CatmullRomChain(alpha, points, nP=1024)
    mask = np.ones((H, W), dtype=np.int8)
    for m, n in c:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 0
    return mask

def spline_multi_mask(points, H, W):
    # print(H,W)
    alpha = 0.5
    # Calculate the Catmull-Rom splines through the points
    c1 = CatmullRomChain(alpha, points[0], nP=1024)
    c2 = CatmullRomChain(alpha, points[1], nP=1024)
    mask = np.ones((H, W), dtype=np.int8)
    for m, n in c1:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 0
    for m, n in c2:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 0
    return mask

def get_mask(points, H, W):
    # print(H,W)
    alpha = 0.5
    # Calculate the Catmull-Rom splines through the points
    c = CatmullRomChain(alpha, points, nP=1024)
    mask = np.zeros((H, W), dtype=np.int8)
    for m, n in c:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 1
    return mask

def get_multi_mask(points, H, W):
    # print(H,W)
    alpha = 0.5
    # Calculate the Catmull-Rom splines through the points
    c1 = CatmullRomChain(alpha, points[0], nP=1024)
    c2 = CatmullRomChain(alpha, points[1], nP=1024)
    mask = np.zeros((H, W), dtype=np.int8)
    for m, n in c1:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 1
    for m, n in c2:
        if m != m or n != n:
            continue
        if m < 0 or m > H - 1 or n < 0 or n > W - 1:
            continue
        else:
            mask[int(m)][int(n)] = 1
    return mask

