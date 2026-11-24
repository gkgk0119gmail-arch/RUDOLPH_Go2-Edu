"""Falls / tracking table of one stage: python scripts/summarize_stage.py s4"""
import json,sys,statistics as S
s=sys.argv[1]
for w in('teacher','student'):
  for st in('','_stress'):
    for d in(0,20,40):
      r=[x for x in json.load(open(f'results/gap_search/eval_{s}_{w}{st}_d{d}.json'))['rows'] if x['height_cmd']>=.14]; mv=[x for x in r if x['speed_cmd']>0]
      F=lambda q:round(sum(x['fell']*x['n'] for x in q))
      print(w+st,d,'fell %d/%d'%(F(r),sum(x['n'] for x in r)),'herr',max(x['height_err'] for x in mv),'tq',max(x['torque_peak'] for x in r),'v.3',round(S.mean(x['speed'] for x in mv if x['speed_cmd']==.3),3),'v.6',round(S.mean(x['speed'] for x in mv if x['speed_cmd']==.6),3),'h',{h:F([x for x in r if x['height_cmd']==h]) for h in(.32,.26,.2,.16,.14)},'v',{v:F([x for x in r if x['speed_cmd']==v]) for v in(0,.3,.6)})
r=[x for x in json.load(open(f'results/gap_search/mujoco_student_{s}.json')) if x['height_cmd']>.13]
for dl in(0,10,20,30,40):
  for ro in(False,True):
    q=[x for x in r if x['delay_ms']==dl and x['rough']==ro]
    print('mujoco',dl,'sill' if ro else 'flat',sum(x['fell'] for x in q),len(q),{h:sum(x['fell'] for x in q if x['height_cmd']==h) for h in(.32,.26,.2,.16)},'v.3',round(S.mean(x['speed'] for x in q if x['speed_cmd']==.3 and not x['fell']),3))

def success(name):  # no fall and >= 60 % of the commanded speed (moving commands only)
  r=[x for x in json.load(open(f'results/gap_search/{name}.json')) if x['height_cmd']>.13 and x['speed_cmd']>0]
  for ro in(False,True):
    for v in(.3,.6):
      print('success',name,'sill' if ro else 'flat',v,{dl:'%d/%d'%(sum(not x['fell'] and x['speed']>=.6*v for x in q),len(q)) for dl in(0,10,20,30,40) for q in[[x for x in r if x['rough']==ro and x['speed_cmd']==v and x['delay_ms']==dl]]})
for n in ('mujoco_student','mujoco_student_s3','mujoco_student_s4') + ((f'mujoco_student_{s}',) if s not in('s3','s4') else ()): success(n)
