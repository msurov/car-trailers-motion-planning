from draw_car_trailer import CarTrailer
from car_trailer_kinematics import CarTrailerKinematic
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib import animation
import matplotlib.pyplot as plt


def anim(traj, cartrailer, filepath=None, fps=30, speedup=1):
    t = traj['t']
    x = traj['x']
    y = traj['y']
    st = np.array([x, y, traj['theta'], traj['psi'], traj['phi']]).T
    sp = make_interp_spline(t, st, k=1)

    fig = plt.figure(figsize=(8,8))
    plt.gca().set_aspect('equal')
    ax = plt.gca()
    plt.grid(True)

    maxx = np.max(x)
    minx = np.min(x)
    maxy = np.max(y)
    miny = np.min(y)
    diapx = maxx - minx
    diapy = maxy - miny
    mx = (maxx + minx) / 2
    my = (maxy + miny) / 2
    diap = max(diapx, diapy) + 2 * cartrailer.cfg.wheel_base
    ax.set_xlim([mx - diap / 2, mx + diap / 2])
    ax.set_ylim([my - diap / 2, my + diap / 2])

    def update(i):
        ti = t[0] + i * speedup / fps
        sti = sp(ti)
        cartrailer.move(*sti)
        return cartrailer.patches()

    def init():
        patches = cartrailer.patches()
        [ax.add_patch(p) for p in patches]
        return patches

    nframes = int((t[-1] - t[0]) * fps / speedup)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    if filepath is not None:
        if filepath.endswith('.gif'):
            writer='imagemagick'
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Maksim Surov'), bitrate=400*60)
        anim.save(filepath, writer)
    else:
        plt.show()


if __name__ == '__main__':
    data = np.load('data/data.npy', allow_pickle=True).item()
    cfg = data['cfg']
    cartrailer = CarTrailer(cfg)
    traj = data['trajectory']
    anim(traj, cartrailer, fps=60, filepath='data/anim.mp4')
    