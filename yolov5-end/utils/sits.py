def xyxyTOxywh(x0,y0,x1,y1):
    x = (x0+x1)/2
    y = (y0+y1)/2
    w = x1-x0
    h = y1-y0
    return x,y,w,h

def sit_2(W,H,loc):
    a = 0.2
    b = 0.1
    (x0,y0,x1,y1) = loc
    x,y,w,h = xyxyTOxywh(x0,y0,x1,y1)
    if (x0>a*W) & (x1<(1-a)*W) & (y0>b*H):
        return True
    else:
        return False