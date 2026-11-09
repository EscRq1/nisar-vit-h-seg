#!/usr/bin/env python3
"""Build a geographically grouped scenes.csv from local NISAR L2 GCOV HDF5 files.

Reads HDF5 map coordinates/projection, not SAR pixels. No manual area labels.
Requires: pip install h5py numpy pyproj shapely

Example:
    python auto_nisar_manifest.py --data ./data --out scenes.csv

Every spatially overlapping or nearby scene is assigned the same group and split.
A separate validation group is required by the supplied nisar_mae.py script.
"""

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon


GEOD = Geod(ellps="WGS84")


@dataclass
class Scene:
    path: Path
    epsg: int
    bounds: tuple[float, float, float, float]
    center_lon: float
    center_lat: float
    radius_m: float
    outline: list[tuple[float, float]]


def outline_points(bounds, segments_per_edge=8):
    x0, y0, x1, y1 = bounds
    corners = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
    points=[]
    for i in range(4):
        ax,ay=corners[i]
        bx,by=corners[(i+1)%4]
        for k in range(segments_per_edge):
            t = k/segments_per_edge
            points.append((ax+(bx-ax)*t, ay+(by-ay)*t))
    return points


def inspect_scene(path, frequency, channels):
    group_path = f"/science/LSAR/GCOV/grids/frequency{frequency}"
    with h5py.File(path, "r") as f:
        if group_path not in f:
            raise ValueError(f"Missing {group_path}")
        grid=f[group_path]
        for ch in channels:
            if ch not in grid or grid[ch].ndim != 2 or grid[ch].dtype.kind != "f":
                raise ValueError(f"Missing/incompatible polarization power channel {ch}")
        for key in ("xCoordinates", "yCoordinates", "xCoordinateSpacing", "yCoordinateSpacing", "projection"):
            if key not in grid:
                raise ValueError(f"Missing {key}")
        nx = len(grid["xCoordinates"])
        ny = len(grid["yCoordinates"])
        if nx < 2 or ny < 2 or grid[channels[0]].shape != (ny,nx):
            raise ValueError("Unexpected coordinate lengths or science array dimensions")
        x0=float(grid["xCoordinates"][0]); x1=float(grid["xCoordinates"][nx-1])
        y0=float(grid["yCoordinates"][0]); y1=float(grid["yCoordinates"][ny-1])
        sx=abs(float(grid["xCoordinateSpacing"][()]))
        sy=abs(float(grid["yCoordinateSpacing"][()]))
        epsg=int(grid["projection"][()])
    if not all(math.isfinite(v) for v in (x0,x1,y0,y1,sx,sy)) or sx<=0 or sy<=0:
        raise ValueError("Invalid spatial coordinate values")
    bounds=(min(x0,x1)-sx/2, min(y0,y1)-sy/2, max(x0,x1)+sx/2, max(y0,y1)+sy/2)
    outline=outline_points(bounds)
    to_wgs=Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    center_lon, center_lat=to_wgs.transform((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2)
    lons,lats=to_wgs.transform([p[0] for p in outline], [p[1] for p in outline])
    if not all(math.isfinite(v) for v in [center_lon,center_lat,*lons,*lats]):
        raise ValueError("Coordinates could not be converted to WGS84")
    _,_,distance=GEOD.inv([center_lon]*len(lons),[center_lat]*len(lats),lons,lats)
    radius=max(distance)
    return Scene(path,epsg,bounds,center_lon,center_lat,radius,outline)


def overlaps(a, b, buffer_m):
    # Geodesic bounding-circle prefilter: if the footprint outlines are far apart,
    # no expensive projected geometry comparison is needed.
    _,_,dist=GEOD.inv(a.center_lon,a.center_lat,b.center_lon,b.center_lat)
    if dist > a.radius_m+b.radius_m+buffer_m:
        return False

    # Compare both geographic footprints in one local meter-based CRS. This is
    # safer than comparing UTM x/y values across different UTM zones.
    local=CRS.from_proj4(
        f"+proj=aeqd +lat_0={a.center_lat} +lon_0={a.center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    def polygon(scene):
        transformer=Transformer.from_crs(CRS.from_epsg(scene.epsg),local,always_xy=True)
        x,y=transformer.transform([p[0] for p in scene.outline],[p[1] for p in scene.outline])
        poly=Polygon(zip(x,y))
        if not poly.is_valid:
            poly=poly.buffer(0)
        return poly
    return polygon(a).distance(polygon(b)) <= buffer_m


def make_groups(scenes, buffer_m):
    parent=list(range(len(scenes)))
    def root(i):
        while parent[i] != i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(i,j):
        ri,rj=root(i),root(j)
        if ri!=rj:
            parent[rj]=ri
    for i,a in enumerate(scenes):
        for j in range(i+1,len(scenes)):
            if overlaps(a,scenes[j],buffer_m):
                union(i,j)
    components=defaultdict(list)
    for i in range(len(scenes)):
        components[root(i)].append(i)
    return sorted(components.values(), key=lambda group: min(str(scenes[i].path) for i in group))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,help="Folder containing downloaded .h5 files (searched recursively)")
    p.add_argument("--out",default="scenes.csv")
    p.add_argument("--frequency",choices=["A","B"],default="A")
    p.add_argument("--channels",nargs=2,default=["HHHH","HVHV"])
    p.add_argument("--val-fraction",type=float,default=0.2,help="Fraction of geographic groups for validation")
    p.add_argument("--buffer-m",type=float,default=1000.0,help="Group scenes separated by <= this distance in meters")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--group-prefix", default="geo", help="Use a new prefix when generating a separate batch, e.g. batch2_geo")
    args=p.parse_args()
    if not 0 < args.val_fraction < 1 or args.buffer_m < 0:
        p.error("Need 0 < --val-fraction < 1 and --buffer-m >= 0")
    files=sorted(Path(args.data).expanduser().resolve().rglob("*.h5"))
    if not files:
        p.error(f"No .h5 files found under {args.data}")
    print(f"Inspecting geographic metadata in {len(files)} HDF5 files...", flush=True)
    scenes=[]
    for i,path in enumerate(files,1):
        try:
            scene=inspect_scene(path,args.frequency,args.channels)
        except Exception as exc:
            raise RuntimeError(f"Cannot inspect {path.name}: {exc}") from exc
        scenes.append(scene)
        print(f"[{i}/{len(files)}] {path.name}: EPSG:{scene.epsg}, "
              f"center≈({scene.center_lat:.3f}, {scene.center_lon:.3f})",flush=True)
    groups=make_groups(scenes,args.buffer_m)
    if len(groups)<2:
        raise SystemExit(
            "ERROR: All scenes form one overlapping geographic group. "
            "A geographically independent train/validation split isn't possible "
            "with these footprints. Add scenes from another location, or create "
            "carefully selected non-overlapping crop windows and adapt grouping. "
            "No CSV was written."
        )
    count_val=max(1,min(len(groups)-1,round(len(groups)*args.val_fraction)))
    ids=list(range(len(groups)))
    random.Random(args.seed).shuffle(ids)
    val_ids=set(ids[:count_val])
    by_idx={}
    for group_id,members in enumerate(groups):
        area=f"{args.group_prefix}_{group_id+1:03d}"
        split="val" if group_id in val_ids else "train"
        for idx in members:
            by_idx[idx]=(area,split)
    out=Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["path","area","split"])
        for i,scene in enumerate(scenes):
            writer.writerow([str(scene.path),*by_idx[i]])
    print(f"\nFound {len(groups)} non-overlapping geographic groups:")
    for g,members in enumerate(groups):
        s=scenes[members[0]]
        print(f"  {args.group_prefix}_{g+1:03d}: {len(members):2d} files, "
              f"{'val' if g in val_ids else 'train':5s}, "
              f"example center {s.center_lat:.3f}, {s.center_lon:.3f}")
    print(f"\nSaved {len(files)} rows to {out}")
    print("NOTE: Grouping is based on scene footprints (including no-data margins), "
          "not on radar image content. Inspect large unexpected groups if needed.")


if __name__=="__main__":
    main()
