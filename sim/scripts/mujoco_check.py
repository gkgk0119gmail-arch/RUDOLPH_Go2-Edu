"""Cross-check the student in MuJoCo (native engine, CPU): height x speed x control latency x friction.

    python scripts/mujoco_check.py --ckpt weights/go2_crouch_student.pt --out results/gap_search/mujoco_student.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "go2_lab"))
from student import History, Student  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{l}_{j}_joint" for j in ("hip", "thigh", "calf") for l in LEGS]  # Isaac Lab joint order
Q0 = np.array([.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5])
TMAX = np.array([23.7] * 8 + [45.43] * 4)
VMAX = np.array([30.1] * 8 + [15.7] * 4)
DT, CTRL = 0.0025, 0.02


def model(friction, rough, seed):
    root = ET.parse(ROOT / "assets/go2_xt16/go2_xt16.urdf").getroot()
    for link in root.iter("link"):
        for v in link.findall("visual"):
            link.remove(v)
        for c in link.findall("collision"):
            if c.find("geometry/mesh") is not None:
                raise SystemExit(f"mesh collider on {link.get('name')}")
    root.insert(0, ET.fromstring('<mujoco><compiler fusestatic="true" discardvisual="true"/></mujoco>'))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    base = spec.worldbody.first_body()
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in base.joints):
        base.add_freejoint()
    for g in spec.geoms:
        g.contype, g.conaffinity, g.friction = 1, 0, [friction, 0.005, 0.0001]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, .1], contype=0,
                            conaffinity=1, friction=[friction, 0.005, 0.0001])
    if rough:
        r = np.random.default_rng(seed)  # 3 cm sills across the path, 1-3 cm high
        for x in np.arange(0.8, 12, 0.6):
            h = r.uniform(0.01, 0.03) / 2
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, 0, h], size=[r.uniform(.03, .15), 5, h],
                                    contype=0, conaffinity=1, friction=[friction, 0.005, 0.0001])
    spec.option.timestep = DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec.compile()


def rollout(a):
    ck, h, v, delay, fr, rough, seed, T = a
    torch.set_num_threads(1)
    net = Student(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["student"]); net.eval()
    m = model(fr, rough, seed); d = mujoco.MjData(m); rng = np.random.default_rng(seed)
    qa = np.array([m.joint(j).qposadr[0] for j in JOINTS]); va = np.array([m.joint(j).dofadr[0] for j in JOINTS])
    d.qpos[2] = 0.36; d.qpos[qa] = Q0 + rng.uniform(-.05, .05, 12); mujoco.mj_forward(m, d)
    hist = History(1, "cpu"); last = np.zeros(12); queue = []; target = Q0.copy()
    R = dict(h=[], v=[], tilt=[], y=[]); fell = False; calf = 0; tq = 0.0; n = 0
    calf_geoms = {g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name or "").endswith(("calf", "thigh"))}
    feet = {g for g in calf_geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE}  # feet may be fused into calves
    bodies_base = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.joint(JOINTS[0]).bodyid[0] - 1}
    floor_ok = calf_geoms - feet
    for k in range(int(T / CTRL)):
        Rm = d.xmat[1].reshape(3, 3)
        w = d.qvel[3:6] + rng.uniform(-.2, .2, 3); g = -Rm[2] + rng.uniform(-.05, .05, 3)
        q = d.qpos[qa] - Q0 + rng.uniform(-.01, .01, 12); qd = d.qvel[va] + rng.uniform(-1.5, 1.5, 12)
        obs = np.concatenate([np.zeros(3), w, g, [v, 0, 0], q, qd, last, [0, 0], [h]])
        with torch.no_grad():
            act = net(hist.push(torch.tensor(obs[None], dtype=torch.float32)))[0][0].numpy().astype(float)
        last = act; queue.append((k * CTRL + delay, Q0 + .25 * act))
        for s in range(int(round(CTRL / DT))):
            now = k * CTRL + s * DT
            while queue and queue[0][0] <= now + 1e-9:
                target = queue.pop(0)[1]
            qq, vv = d.qpos[qa], d.qvel[va]
            tau = 25 * (target - qq) - .5 * vv
            hi = np.clip(TMAX * (1 - vv / VMAX), 0, TMAX); lo = np.clip(TMAX * (-1 - vv / VMAX), -TMAX, 0)
            tau = np.clip(tau, lo, hi)  # DC motor
            d.qfrc_applied[va] = tau; mujoco.mj_step(m, d)
            tq = max(tq, float(np.abs(tau / TMAX).max())) if now > 2 else tq
        up = d.xmat[1][8]
        touch = {int(x) for c in d.contact[:d.ncon] for x in c.geom}
        if up < 0.5 or touch & bodies_base:
            if k * CTRL > 0.5:
                fell = True; break
        if k * CTRL > 3:
            n += 1; calf += bool(touch & floor_ok)
            R["h"].append(d.qpos[2]); R["v"].append(float(d.xmat[1].reshape(3, 3)[:, 0] @ d.qvel[:3]))
            R["tilt"].append(np.degrees(np.arccos(min(1, up))))
    f = lambda x: float(np.mean(x)) if x else None
    return dict(height_cmd=h, speed_cmd=v, delay_ms=round(delay * 1e3), friction=fr, rough=rough, seed=seed, fell=fell,
                height=f(R["h"]), speed=f(R["v"]), tilt=f(R["tilt"]), leg_contact=calf / n if n else None, torque_peak=tq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    J = [(a.ckpt, h, v, dl, fr, ro, s, a.seconds) for h in (.32, .26, .2, .16, .12) for v in (0., .3, .6)
         for dl in (0., .005, .01, .02, .03, .04) for fr in (.4, .8) for ro in (False, True) for s in range(a.seeds)]
    m = model(.8, False, 0); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    print("total mass", m.body_mass.sum(), "bodies", m.nbody, "geoms", m.ngeom, "jobs", len(J), flush=True)
    with Pool(32) as p:
        rows = p.map(rollout, J, chunksize=4)
    Path(a.out).write_text(json.dumps(rows))
    for dl in sorted({r["delay_ms"] for r in rows}):
        for ro in (False, True):
            q = [r for r in rows if r["delay_ms"] == dl and r["rough"] == ro]; ok = [r for r in q if not r["fell"]]
            print(f"delay {dl} rough {ro} fell {sum(r['fell'] for r in q)}/{len(q)}", "herr %.3f" % np.mean([abs(r["height"] - r["height_cmd"]) for r in ok]),
                  "verr %.3f" % np.mean([abs(r["speed"] - r["speed_cmd"]) for r in ok]), "leg %.3f" % np.mean([r["leg_contact"] for r in ok]),
                  "tq %.2f" % np.mean([r["torque_peak"] for r in ok]))
