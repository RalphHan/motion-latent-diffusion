from .quaternion import *

offsets = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0],
                    [0, -1, 0]])

tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]


def get_rotation_naive(sons, joints):
    ret = np.zeros(joints.shape[:-2] + (4,))
    for j in sons[1:]:
        u = offsets[j][np.newaxis, ...].repeat(len(joints), axis=0)
        v = joints[:, j] - joints[:, sons[0]]
        v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
        ret += qbetween_np(u, v)
    # ret /= len(sons) - 1
    ret /= np.sqrt((ret ** 2).sum(axis=-1, keepdims=True))
    return ret


def ik(joints):
    roots = []
    sons = [[0, 1, 2, 3], [9, 12, 13, 14]]
    for i in range(len(sons)):
        roots.append(get_rotation_naive(sons[i], joints))

    quat_params = np.zeros(joints.shape[:-1] + (4,))
    quat_params[..., 0] = 1
    quat_params[:, 0] = roots[0]
    for chain in tree:
        R = roots[0] if chain[0] == 0 else roots[1]
        for j in range(len(chain) - 1):
            if chain[j] == 0: continue
            if chain[0] == 0 and chain[j] == 9:
                quat_params[:, 9] = qmul_np(qinv_np(R), roots[1])
                R = roots[1]
                continue
            if chain[j] == 9: continue
            u = offsets[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
            v = joints[:, chain[j + 1]] - joints[:, chain[j]]
            v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
            rot_u_v = qbetween_np(u, v)
            quat_params[:, chain[j]] = qmul_np(qinv_np(R), rot_u_v)
            R = qmul_np(R, quat_params[:, chain[j]])

    return quat_params, joints[..., 0, :]


def fk(rotations, root_positions):
    rotations = torch.Tensor(rotations)
    root_positions = torch.Tensor(root_positions)
    _offsets = torch.Tensor(offsets)
    _parents = np.array(parents)

    _has_children = np.zeros(len(_parents)).astype(bool)
    for i, parent in enumerate(_parents):
        if parent != -1:
            _has_children[parent] = True

    _children = [[] for _ in _parents]
    for i, parent in enumerate(_parents):
        if parent != -1:
            _children[parent].append(i)

    positions_world = []
    rotations_world = []

    expanded_offsets = _offsets.expand(
        rotations.shape[0],
        _offsets.shape[0],
        _offsets.shape[1],
    )

    # Parallelize along the batch and time dimensions
    for i in range(_offsets.shape[0]):
        if _parents[i] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, 0])
        else:
            positions_world.append(
                qrot(
                    rotations_world[_parents[i]], expanded_offsets[:, i]
                )
                + positions_world[_parents[i]]
            )
            if _has_children[i]:
                rotations_world.append(
                    qmul(
                        rotations_world[_parents[i]], rotations[:, i]
                    )
                )
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=1).numpy()


if __name__ == '__main__':
    import json
    import binascii
    with open("playground/f5ebdd69-6bc5-493b-b808-72ef24b72916.json") as f:
        data = json.load(f)
    joints = np.frombuffer(binascii.a2b_base64(data["positions"]), dtype=data["dtype"]).reshape(data["n_frames"],
                                                                                                data["n_joints"], 3)
    quat, root_pos = ik(joints)
    with open(f"playground/angle.json", "w") as f:
        json.dump({"root_positions": binascii.b2a_base64(
            root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                   "rotations": binascii.b2a_base64(quat.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                   "dtype": "float32",
                   "fps": data["fps"],
                   "mode": "quaternion",
                   "n_frames": data["n_frames"],
                   "n_joints": data["n_joints"]}, f, indent=4)

    new_joints = fk(quat, root_pos)
    from mld.data.humanml.utils.plot_script import plot_3d_motion

    plot_3d_motion("playground/joints.mp4", joints * 1.3, radius=3, title="", fps=data["fps"])
    plot_3d_motion(f"playground/new_joints.mp4", new_joints * 1.3, radius=3,
                   title="", fps=data["fps"])
