from os.path import join as join_path
from os.path import sep as os_sep


A2D2_PATH = "/home/g.leontiev/a2d2/"
ss_p = join_path(A2D2_PATH, "camera_lidar_semantic")
sens_ext = {
    "camera": ".png",
    "label": ".png",
    "lidar": ".npz"
}

rel_ = lambda __p: join_path(*__p.split(os_sep)[__p.split(os_sep).index('camera_lidar_semantic'):])
abs_ = lambda __p: join_path(A2D2_PATH, __p)


def sensor_p(_id, s_type):
    if s_type not in sens_ext.keys(): raise ValueError("Wrong sensor type: s_type")
    d,t,s = _id.split("_")
    _p = "_".join([d, s_type, s, t]) + sens_ext[s_type]
    _p = join_path(ss_p, f"{d[:8]}_{d[8:]}", s_type, f"cam_{sa_(s)}", _p)
    return rel_(_p)


def sa_(x):
    als = ["center", "left", "right"]
    for o in als:
        if o in x:
            return x.replace(o, "_" + o)
    raise ValueError(f"Bad index contains wrong sensor align: {x}")
