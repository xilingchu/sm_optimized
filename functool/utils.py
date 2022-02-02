import random

def randomRange(lb, rb):
    _center = (lb + rb)/2
    _ratio  = abs(rb - lb)
    return _ratio*(random.random() - (0.5 - _center))

if __name__ == '__main__':
    print(randomRange(-2, 2))
