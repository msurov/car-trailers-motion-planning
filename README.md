The flatness-based method for motion planning of car-trailers system

![alt text](img/anim.gif)

The script gives a trivial implementation of the flatness-based motion planning for the car N-trailers system. 

# Point-to-point trajectory
can be found with the help of the `trajectory_2pts(initial_state, final_state)` function, where the initial and 
final state arguments are composed of the variables `state = [x,y,phi,theta0,theta1,...,thetaN]`; 
`x,y` are the first trailer Cartesian coordinates; `phi` is the steering angle; `theta0,...,thetaN` are orientations of the trailers wrt the world frame. 

## Example
The car starts moving from the origin and stops at the point with coordinates `(-7,2)`. All the trailers are aligned along a vertical line.

```python
  st1 = [0, 0, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
  st2 = [7, -2, 0, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]
  traj = trajectory_2pts(st1, st2)
```

# A trajectory through several Cartesian waypoints
for `ntrailers` system can be found with the help of the `trajectory_through_waypoints(waypts, ntrailers)` function, where the waypoints formed as
```python
waypts = [
        [x1, y1],
        [x2, y2],
        ...
        [xN, yN],
    ].
```

## Example
It is assumed that the car moves through the five zigzag shaped points.
```python
    waypts = [
        [0, 0],
        [5, 5],
        [10, 0],
        [15, 5],
        [20, 0],
    ]
    traj = trajectory_through_waypoints(waypts, 3)
```
