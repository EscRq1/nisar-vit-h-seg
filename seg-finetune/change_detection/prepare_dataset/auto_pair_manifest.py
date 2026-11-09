#!/usr/bin/env python3
"""Match NISAR GCOV filenames in <area>_t0 / <area>_t1 directories."""
import argparse, csv, re, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PAT = re.compile(r'^NISAR_L2_PR_GCOV_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<direction>[AD])_(?P<frame>\d{3})_(?P<mode>\d{4})_(?P<polarization>[A-Z0-9]+)_(?P<swath>[A-Z])_(?P<start>\d{8}T\d{6})_\d{8}T\d{6}_.*\.h5$',re.I)
FIELDS=['pair_id','area','split','t0_date','t1_date','interval_days','track','direction','frame','mode','polarization','swath','t0_path','t1_path']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,required=True,help='Parent containing area_t0 and area_t1 folders')
 p.add_argument('--out',type=Path,default=Path('scene_pairs.csv'))
 a=p.parse_args();root=a.data.resolve()
 if not root.is_dir():p.error(f'Data directory not found: {root}')
 if a.out.exists():p.error(f'Output exists: {a.out}; choose another filename')
 groups=defaultdict(lambda:[[],[]]);issues=[]
 folders=sorted(x for x in root.iterdir() if x.is_dir() and re.fullmatch(r'.+_t[01]',x.name,re.I))
 if not folders:p.error('No <area>_t0 or <area>_t1 directories found directly under --data')
 for folder in folders:
  area=folder.name[:-3];time=int(folder.name[-1])
  if not re.fullmatch(r'[A-Za-z0-9_-]+',area):issues.append(f'Invalid area name: {folder}');continue
  for path in sorted(folder.rglob('*.h5')):
   m=PAT.fullmatch(path.name)
   if not m:issues.append(f'Unrecognized filename: {path}');continue
   d=m.groupdict();key=(area,*(d[x] for x in ('track','direction','frame','mode','polarization','swath')))
   try:dt=datetime.strptime(d['start'],'%Y%m%dT%H%M%S')
   except ValueError:issues.append(f'Invalid date: {path}');continue
   groups[key][time].append((path.resolve(),dt))
 rows=[]
 for key,(zero,one) in sorted(groups.items()):
  area,track,direction,frame,mode,pol,swath=key
  tag='_'.join(key)
  if len(zero)!=1 or len(one)!=1:
   issues.append(f'{tag}: {len(zero)} t0 and {len(one)} t1 files; skipped (ambiguous or unmatched). '+ '; '.join(str(x[0]) for x in zero+one));continue
  (f0,t0),(f1,t1)=zero[0],one[0]
  if t1<=t0:issues.append(f'{tag}: t1 is not later than t0; skipped');continue
  # Retain prior pair-id convention: mode here corresponds to the four-digit filename field.
  pid='_'.join((area,track,direction,frame,mode,swath))
  rows.append(dict(pair_id=pid,area=area,split='',t0_date=t0.isoformat(),t1_date=t1.isoformat(),interval_days=(t1-t0).total_seconds()/86400,track=track,direction=direction,frame=frame,mode=pol,polarization=swath,swath=swath,t0_path=str(f0),t1_path=str(f1)))
 if len({r['pair_id'] for r in rows})!=len(rows):p.error('Pair IDs collide; no CSV written')
 if not rows:
  for issue in issues:print('WARNING:',issue,file=sys.stderr)
  p.error('No unambiguous pairs found; no CSV written')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 for issue in issues:print('WARNING:',issue,file=sys.stderr)
 print(f'Found {len(rows)} pairs; saved {a.out.resolve()}')
 print(f'{len(issues)} warnings. Pairing uses filename metadata only; verify HDF5 channels, geographic overlap and registration separately. split left blank.')
 if issues:sys.exit(1)
if __name__=='__main__':main()
