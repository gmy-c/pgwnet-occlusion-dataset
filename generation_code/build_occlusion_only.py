"""Occlusion-only entry point extracted from the hash-verified historical builder.
Sampling/compositing follows the historical builder; H includes 0.65 per the manuscript.
This adapter is newly prepared and is not a hash-verified historical source.
"""
from pathlib import Path
import sys, argparse, importlib.util, json, copy, shutil
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent/'original_runtime'))
import cv2
import numpy as np
from pycocotools import mask as mu
from lite_operators import VERSION,LEVELS,seed_for
from build_lite import get_rgb,dump

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ann',required=True,help='Original 998-image BDD8 validation COCO JSON')
    ap.add_argument('--images',required=True,help='Directory containing all original validation images, including donors')
    ap.add_argument('--out',required=True,help='New output directory; existing directories are refused')
    a=ap.parse_args();out=Path(a.out).resolve()
    if out.exists():raise RuntimeError('Choose a new output directory')
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('legacy_builder',here/'legacy/tools/build_protocol.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    val=old.Source(a.ann,a.images)
    assert len(val.images)==998,'Use the original 998-image validation subset'
    # Localize absolute paths in the source annotation without changing IDs or annotations.
    for im in val.images.values():
        im['file_name']=str((Path(a.images)/Path(im['file_name']).name).resolve())
    frozen=json.loads((here/'partitions.json').read_text())
    parts=frozen['legacy_source_partitions'];donor_pools=frozen['donor_pools']
    for pool in donor_pools.values():
        assert all(i in val.images for i in pool)
    out.mkdir();(out/'annotations').mkdir();(out/'manifests').mkdir()
    cv2.setNumThreads(1)
    def progress(stage,**kw):print(stage,kw,flush=True)
    counts={};reject=Counter();occsets={};total_occ_checks=0
    def make_triplet(anchor,donors,tag):
        i,aid,cat=anchor;im=val.images[i];h,w=im['height'],im['width'];anns=val.anns[i];masks=val.masks(i);target=masks[next(j for j,v in enumerate(anns) if v['id']==aid)];yy,xx=np.where(target)
        target_area=int(target.sum())
        if target_area<128:return None
        rng=np.random.RandomState(seed_for(tag,aid,'fixed_donor_trajectory'))
        for attempt in range(100):
            di,da,dc=donors[int(rng.randint(len(donors)))];danns=val.anns[di];dj=next(j for j,v in enumerate(danns) if v['id']==da);dm=val.masks(di)[dj];dy,dx=np.where(dm)
            if len(dx)<128:continue
            crop=get_rgb(val.images[di]['file_name'])[dy.min():dy.max()+1,dx.min():dx.max()+1];cut=dm[dy.min():dy.max()+1,dx.min():dx.max()+1]
            scale=float(np.clip((yy.max()-yy.min()+1)/cut.shape[0]*rng.uniform(1.05,1.8),.15,3.))
            dh=max(2,round(cut.shape[0]*scale));dw=max(2,round(cut.shape[1]*scale))
            if dh>h//2 or dw>w//2:continue
            small=cv2.resize(cut.astype(np.uint8),(dw,dh),interpolation=cv2.INTER_NEAREST).astype(bool);pixels=cv2.resize(crop,(dw,dh),interpolation=cv2.INTER_LINEAR)
            y=int(np.clip(yy.max()+1-dh+rng.uniform(-.05,.10)*(yy.max()-yy.min()+1),0,h-dh));lo=max(0,int(xx.min()-dw));hi=min(w-dw,int(xx.max()));positions=np.unique(np.linspace(lo,hi,100).astype(int))
            choices={};last_x=-1
            for lev in LEVELS:
                bounds=old.LEVELS[lev]
                for x in positions:
                    if x<last_x:continue
                    ratio=float(np.count_nonzero(target[y:y+dh,x:x+dw]&small)/target_area)
                    if bounds[0] <= ratio and (ratio <= bounds[1] if lev == 'H' else ratio < bounds[1]) and target_area*(1-ratio)>=64:
                        choices[lev]=(int(x),ratio);last_x=int(x)+1;break
                if lev not in choices:break
            if len(choices)==3:
                results={}
                for lev,(x,ratio) in choices.items():
                    occ=np.zeros_like(target);occ[y:y+dh,x:x+dw]=small;rgb=get_rgb(im['file_name']);patch=rgb[y:y+dh,x:x+dw];patch[small]=pixels[small]
                    changed=[];new=[]
                    for ann,mask in zip(anns,masks):
                        remaining=mask&~occ;changed.append({'source_ann_id':ann['id'],'old_area':int(mask.sum()),'new_area':int(remaining.sum())})
                        if not remaining.any():continue
                        z=copy.deepcopy(ann);z['segmentation']=old.encode(remaining);z['area']=int(remaining.sum());z['bbox']=mu.toBbox(z['segmentation']).tolist();assert np.array_equal(old.decode(z,h,w),remaining);new.append(z)
                    new.append(dict(id=-1,image_id=i,category_id=dc,segmentation=old.encode(occ),area=int(occ.sum()),bbox=mu.toBbox(old.encode(occ)).tolist(),iscrowd=0,ignore=0))
                    rec=dict(protocol=VERSION,source_image_id=i,source_file=im['file_name'],target_ann_id=aid,target_category_id=cat,donor_image_id=di,donor_ann_id=da,donor_split=tag,scale=scale,x=x,y=y,trajectory='fixed_asset_left_entry',family='foreground_occlusion',level=lev,added_occlusion_ratio=ratio,original_visible_target=old.encode(target),occluder=old.encode(occ),changes=changed)
                    results[lev]=(rgb,new,rec)
                return results
        return None
    for split,pool,limit in [('dev',parts['dev'],60),('report',parts['locked_test'],300)]:
        donors=val.candidates(donor_pools[split]);writers={l:old.DatasetWriter(val,out,f'val_occlusion_{split}_{l}') for l in LEVELS};accepted=[]
        for anchor in old.anchors(val,pool,len(pool)):
            results=make_triplet(anchor,donors,split)
            if results is None:reject[split+':no_shared_donor_trajectory']+=1;continue
            accepted.append(anchor)
            for lev,(rgb,anns,rec) in results.items():
                # Legacy DatasetWriter expects BGR pixels; labels are unchanged.
                writers[lev].add(anchor,(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),anns,rec));total_occ_checks+=len(val.anns[anchor[0]])
            if len(accepted)%20==0:progress('paired_occlusion',split=split,accepted=len(accepted),target=limit)
            if len(accepted)==limit:break
        assert len(accepted)==limit,(split,len(accepted),reject)
        for lev,writer in writers.items():
            recs=writer.finish();counts[f'val_occlusion_{split}_{lev}']=len(recs);occsets[(split,lev)]=recs
        dump(out/'annotations'/f'val_occlusion_{split}_clean.json',val.export([x[0] for x in accepted]))
    for split in ['dev','report']:
        name=f'val_occlusion_{split}_clean'
        p=out/'annotations'/f'{name}.json';data=json.loads(p.read_text())
        for im in data['images']:
            src=Path(im['file_name']);dst=out/'images'/name/src.name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            im['file_name']=str(dst)
        dump(p,data)
    dump(out/'partitions.json',frozen)
    dump(out/'OCCLUSION_BUILD.json',{'counts':counts,'rejections':dict(reject),'note':'New occlusion-only adapter; compare generated outputs against archived manifests before claiming identical reproduction.'})

if __name__=='__main__':main()
