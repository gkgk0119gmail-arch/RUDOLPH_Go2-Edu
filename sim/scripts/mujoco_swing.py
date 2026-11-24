"""Peak foot lift [m] per foot (FL FR RL RR) on flat ground in MuJoCo: checkpoints as arguments."""
import sys; sys.path.insert(0,'scripts')
import mujoco_check as c, numpy as np, mujoco, torch
def peak(ck,h,v):
    net=c.Student(); net.load_state_dict(torch.load(ck,weights_only=True)['student']); m=c.model(.8,False,0); d=mujoco.MjData(m)
    qa=[m.joint(j).qposadr[0] for j in c.JOINTS]; va=[m.joint(j).dofadr[0] for j in c.JOINTS]
    d.qpos[2]=.36; d.qpos[qa]=c.Q0; mujoco.mj_forward(m,d); hi=c.History(1,'cpu'); last=np.zeros(12)
    feet=[g for g in range(m.ngeom) if m.geom_type[g]==2 and m.body(m.geom_bodyid[g]).name.endswith('calf')]; Z=[]
    for k in range(400):
        R=d.xmat[1].reshape(3,3); o=np.concatenate([np.zeros(3),d.qvel[3:6],-R[2],[v,0,0],d.qpos[qa]-c.Q0,d.qvel[va],last,[0,0],[h]])
        with torch.no_grad(): last=net(hi.push(torch.tensor(o[None],dtype=torch.float32)))[0][0].numpy().astype(float)
        t=c.Q0+.25*last
        for _ in range(8):
            d.qfrc_applied[va]=np.clip(25*(t-d.qpos[qa])-.5*d.qvel[va],-c.TMAX,c.TMAX); mujoco.mj_step(m,d)
        if k>150: Z.append(d.geom_xpos[feet,2]-.022)
    return np.array(Z).max(0).round(3)
for ck in sys.argv[1:]:
    for h in(.32,.2,.16): print(ck,h,peak(ck,h,.3),peak(ck,h,.6))
