import numpy as np
from flatness_planner import trajectory_2pts, trajectory_through_waypoints
from animate import animate


def main():
    st1 = [0, 0, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
    st2 = [7, -2, 0, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]
    traj = trajectory_2pts(st1, st2)
    animate(traj, animtime=1)

if __name__ == "__main__":
    main()
