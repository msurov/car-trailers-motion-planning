from dataclasses import dataclass
import numpy as np

@dataclass
class CarTrailerKinematic:
    wheel_base : float # distance between the car front wheel axis
                       #  and the rear wheel axis
    trailer_base : float # distance between the trailer pivot point
                         #  and the trailer wheel axis
    pivot_distance : float # distance between the car rear axis 
                           #  and the trailer pivot point
                           # this parameter is positive is the pivot 
                           # is to the back of the rear wheels
    wheels_diastance : float # distance between left and right wheels

def get_car_pose(q : np.ndarray):
    q = np.array(q)
    x0 = q.T[0]
    y0 = q.T[1]
    theta = q.T[2]
    return np.array([x0, y0, theta]).T


def get_trailer_pose(cfg : CarTrailerKinematic, q : np.ndarray):
    R'''
        # Arguments
        `x`,`y` are the cartesian position of the rear axis central point
        `theta` is the car orientation (CCW)
        `phi` is the steering angle (CCW wrt car frame)
        `psi` is the trailer orientation (CCW art car frame)
    '''
    q = np.array(q)
    x0 = q.T[0]
    y0 = q.T[1]
    theta = q.T[2]
    phi = q.T[3]
    psi = q.T[4]
    d = cfg.pivot_distance
    L2 = cfg.trailer_base

    ct = np.cos(theta)
    st = np.sin(theta)
    theta_psi = theta + psi
    ctp = np.cos(theta_psi)
    stp = np.sin(theta_psi)
    x1 = x0 - d * ct
    y1 = y0 - d * st
    x2 = x1 - L2 * ctp
    y2 = y1 - L2 * stp
    return np.array([x2, y2, theta_psi]).T


if __name__ == '__main__':
    cfg = CarTrailerKinematic(
        2.0, 1.3, 0.4
    )
    q = [
        [0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0],
        [1, 2, 0, 0, 1]
    ]
    print(get_trailer_pose(cfg, q))
