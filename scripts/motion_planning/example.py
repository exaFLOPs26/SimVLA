import torch
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# SE(3) INTERPOLATION UTILS
# ---------------------------

def interpolate_se3(start, end, num_steps):
    """
    Interpolate SE(3) poses with linear-position + SLERP-quaternion.
    start, end: tensors shape (7,) [x,y,z, qw, qx, qy, qz]
    Returns: tensor (num_steps, 7)
    """
    # 1) Positions: simple linear
    p0, p1 = start[:3], end[:3]
    positions = torch.stack([
        p0 + (p1 - p0) * (i / (num_steps - 1))
        for i in range(num_steps)
    ])

    # 2) Quaternions: use SciPy Slerp
    rot0 = R.from_quat([start[4].item(), start[5].item(), start[6].item(), start[3].item()])
    rot1 = R.from_quat([end[4].item(),   end[5].item(),   end[6].item(),   end[3].item()])
    key_times = [0.0, 1.0]
    key_rots = R.concatenate([rot0, rot1])
    slerp = Slerp(key_times, key_rots)
    t_vals = torch.linspace(0.0, 1.0, num_steps).tolist()
    interp_rots = slerp(t_vals)

    quats = torch.stack([
        torch.tensor([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]])
        for r in interp_rots
    ])

    return torch.cat([positions, quats], dim=1)

# ---------------------------
# LIE-ALGEBRA DELTA UTILS
# ---------------------------

def se3_log_delta(q1, q2):
    """
    Compute rotation vector (axis-angle) from q1->q2.
    q1,q2: tensor length-4 [qw,qx,qy,qz]
    returns: rotvec (3,)
    """
    r1 = R.from_quat([q1[1].item(), q1[2].item(), q1[3].item(), q1[0].item()])
    r2 = R.from_quat([q2[1].item(), q2[2].item(), q2[3].item(), q2[0].item()])
    rel = r1.inv() * r2
    return torch.from_numpy(rel.as_rotvec())

# ---------------------------
# TRAJECTORY OPTIMIZATION
# ---------------------------

def optimize_trajectory_se3(
    start, goal,
    num_steps=20,
    iters=300,
    lr=1e-2,
    w_vel=1.0,
    w_acc=0.5,
    collision_fn=None
):
    """
    Optimize an SE(3) trajectory from `start` to `goal`.

    Args:
      start, goal: tensor (7,) = [x,y,z, qw,qx,qy,qz]
      Returns:
        traj: tensor (num_steps,7)
        deltas: tensor (num_steps-1,6) each [dp(3), rotvec(3)]
    """
    traj = interpolate_se3(start, goal, num_steps).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([traj], lr=lr)

    for _ in range(iters):
        optimizer.zero_grad()
        vel_loss = 0.0
        for t in range(num_steps - 1):
            dp = traj[t+1, :3] - traj[t, :3]
            dr = se3_log_delta(traj[t, 3:], traj[t+1, 3:])
            vel_loss = vel_loss + dp.pow(2).sum() + dr.pow(2).sum()
        acc_loss = 0.0
        for t in range(1, num_steps - 1):
            ddp = traj[t+1, :3] - 2*traj[t, :3] + traj[t-1, :3]
            acc_loss = acc_loss + ddp.pow(2).sum()
        coll_loss = collision_fn(traj) if collision_fn else torch.tensor(0.0)
        loss = w_vel * vel_loss + w_acc * acc_loss + coll_loss
        loss.backward()
        with torch.no_grad():
            traj.data[0] = start
            traj.data[-1] = goal
        optimizer.step()

    deltas = []
    for t in range(num_steps - 1):
        dp = traj[t+1, :3] - traj[t, :3]
        dr = se3_log_delta(traj[t, 3:], traj[t+1, 3:])
        deltas.append(torch.cat([dp, dr], dim=0))
    deltas = torch.stack(deltas)

    return traj.detach(), deltas.detach()

# ---------------------------
# VISUALIZATION & SAVE
# ---------------------------

def visualize_and_save(traj, filename='trajectory.png'):
    """
    Plot 3D SE(3) trajectory and save as an image.
    traj: tensor (N,7)
    filename: output PNG file path
    """
    pos = traj[:, :3].numpy()
    quats = traj[:, 3:].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2], '-o', label='Trajectory', alpha=0.7)

    for p, q in zip(pos, quats):
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        dir_vec = r.apply([1, 0, 0])
        ax.quiver(p[0], p[1], p[2],
                  dir_vec[0], dir_vec[1], dir_vec[2],
                  length=0.05, normalize=True, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SE(3) Trajectory')
    ax.legend()
    plt.tight_layout()
    # Save to file
    fig.savefig(filename, dpi=300)
    plt.close(fig)

# ---------------------------
# EXAMPLE USAGE
# ---------------------------
if __name__ == '__main__':
    start = torch.tensor([0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0])
    goal  = torch.tensor([0.3, 0.1, 0.3, 0.9239, 0.0, 0.0, 0.3827])
    traj, deltas = optimize_trajectory_se3(start, goal, num_steps=20, iters=200)
    print('Trajectory shape:', traj.shape)
    print('Deltas shape:', deltas.shape)
    visualize_and_save(traj, filename='se3_trajectory.png')
    print('Saved figure to se3_trajectory.png')

