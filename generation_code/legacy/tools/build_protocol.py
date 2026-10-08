"""BDD visible-instance occlusion protocol, v1.0.

All partitions and placements are deterministic hashes of source identifiers.
This is dataset reproducibility, not repeated model-seed experimentation.
"""
import argparse
import copy
import csv
import hashlib
import html
import json
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_utils

VERSION='bdd-visible-occ-v1.0'
LEVELS={'L':(0.10,0.25),'M':(0.25,0.45),'H':(0.45,0.65)}


def key(*parts):
    return int(hashlib.sha256(('|'.join(map(str,parts))).encode()).hexdigest()[:16],16)


def rng_for(*parts):
    return np.random.RandomState(key(VERSION,*parts) % (2**32))


def encode(mask):
    r=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r['counts']=r['counts'].decode('ascii')
    return r


def decode(ann,h,w):
    s=ann['segmentation']
    if isinstance(s,list):
        if not s:return np.zeros((h,w),bool)
        r=mask_utils.merge(mask_utils.frPyObjects(s,h,w))
    elif isinstance(s['counts'],list):r=mask_utils.frPyObjects(s,h,w)
    else:r=s
    m=mask_utils.decode(r)
    return (m.any(2) if m.ndim==3 else m).astype(bool)


@lru_cache(maxsize=12)
def read_image(path):
    im=cv2.imread(path)
    if im is None:raise FileNotFoundError(path)
    return im


class Source:
    def __init__(self,ann_path,img_dir):
        self.path=Path(ann_path);self.data=json.loads(self.path.read_text())
        self.images={i['id']:dict(i,file_name=str((Path(img_dir)/i['file_name']).resolve())) for i in self.data['images']}
        self.anns=defaultdict(list)
        for a in self.data['annotations']:self.anns[a['image_id']].append(a)
        self.categories=self.data['categories']

    @lru_cache(maxsize=16)
    def masks(self,i):
        im=self.images[i]
        return [decode(a,im['height'],im['width']) for a in self.anns[i]]

    def candidates(self,ids):
        out=[]
        for i in ids:
            for a in self.anns[i]:
                x,y,w,h=a['bbox']
                if not a.get('iscrowd',0) and not a.get('ignore',0) and a['area']>=128 and min(w,h)>=8:
                    out.append((i,a['id'],a['category_id']))
        return out

    def export(self,ids):
        return dict(images=[self.images[i] for i in ids],annotations=[a for i in ids for a in self.anns[i]],categories=self.categories)


def anchors(source,ids,limit):
    """One target per image, class-balanced round robin with unique images."""
    bycat=defaultdict(list)
    for item in source.candidates(ids):bycat[item[2]].append(item)
    for c in bycat:bycat[c].sort(key=lambda x:key('anchor',*x))
    selected=[];used=set();positions={c:0 for c in bycat}
    while len(selected)<limit:
        changed=False
        for c in sorted(bycat):
            while positions[c]<len(bycat[c]) and bycat[c][positions[c]][0] in used:positions[c]+=1
            if positions[c]<len(bycat[c]):
                v=bycat[c][positions[c]];positions[c]+=1;selected.append(v);used.add(v[0]);changed=True
                if len(selected)>=limit:break
        if not changed:break
    return selected


def geometry(target,desired,rng):
    """Connected edge occlusion; opaque non-class texture, not target inpainting."""
    ys,xs=np.where(target);h,w=target.shape
    side=int(rng.randint(4));occ=np.zeros_like(target)
    if side<2:
        values=xs if side==0 else w-1-xs
        cut=int(np.quantile(values,desired))
        left,right=(0,cut+1) if side==0 else (w-1-cut,w)
        occ[max(0,ys.min()-2):min(h,ys.max()+3),left:right]=True
    else:
        values=ys if side==2 else h-1-ys
        cut=int(np.quantile(values,desired))
        top,bottom=(0,cut+1) if side==2 else (h-1-cut,h)
        occ[top:bottom,max(0,xs.min()-2):min(w,xs.max()+3)]=True
    # Restrict the support near the object: avoid a stripe across the whole scene.
    region=np.zeros_like(target);region[max(0,ys.min()-8):min(h,ys.max()+9),max(0,xs.min()-8):min(w,xs.max()+9)]=True
    return occ & region


def copy_occluder(source,target,anchor,donors,desired,rng,bounds):
    h,w=target.shape;ty,tx=np.where(target)
    th=ty.max()-ty.min()+1;tw=tx.max()-tx.min()+1
    for attempt in range(100):
        donor=donors[int(rng.randint(len(donors)))];di,da,_=donor
        if di==anchor[0]:continue
        anns=source.anns[di];idx=next(j for j,a in enumerate(anns) if a['id']==da)
        dm=source.masks(di)[idx];ys,xs=np.where(dm)
        if len(xs)<128:continue
        crop=read_image(source.images[di]['file_name'])[ys.min():ys.max()+1,xs.min():xs.max()+1]
        mask=dm[ys.min():ys.max()+1,xs.min():xs.max()+1]
        scale=np.clip((th/mask.shape[0])*rng.uniform(0.65,1.6),0.15,3.0)
        dh=max(2,round(mask.shape[0]*scale));dw=max(2,round(mask.shape[1]*scale))
        if dh>h//2 or dw>w//2:continue
        small=cv2.resize(mask.astype('uint8'),(dw,dh),interpolation=cv2.INTER_NEAREST).astype(bool)
        # Similar ground-contact height, displaced horizontally across the target.
        bottom=int(ty.max()+rng.uniform(-0.10,0.25)*th)
        y=int(np.clip(bottom-dh,0,h-dh))
        x=int(np.clip(rng.uniform(tx.min()-dw*0.8,tx.max()),0,w-dw))
        overlap=(target[y:y+dh,x:x+dw]&small).sum()/target.sum()
        if bounds[0]<=overlap<bounds[1] and abs(overlap-desired)<=0.10:
            occ=np.zeros_like(target);occ[y:y+dh,x:x+dw]=small
            pixels=cv2.resize(crop,(dw,dh),interpolation=cv2.INTER_LINEAR)
            return occ,(x,y,pixels,small),anns[idx],dict(donor_image_id=di,donor_ann_id=da,scale=float(scale),x=x,y=y)
    return None


def synthesize(source,anchor,donors,family,level,tag):
    i,aid,cat=anchor;im=source.images[i];anns=source.anns[i];masks=source.masks(i)
    j=next(j for j,a in enumerate(anns) if a['id']==aid);target=masks[j]
    if target.sum()<128:return None,'target_area'
    rng=rng_for(tag,aid,family,level);lo,hi=LEVELS[level];desired=float(rng.uniform(lo+0.015,hi-0.015))
    pixels=read_image(im['file_name']).copy();meta={};donor_ann=None
    if family=='geometry':
        occ=geometry(target,desired,rng)
        # Plain smooth background texture has no unlabelled target-class object.
        color=np.median(pixels[~np.logical_or.reduce(masks)],axis=0) if masks and (~np.logical_or.reduce(masks)).any() else np.array([114,114,114])
        texture=np.clip(color+rng.normal(0,2,pixels.shape),0,255).astype(np.uint8)
        pixels[occ]=texture[occ]
    else:
        result=copy_occluder(source,target,anchor,donors,desired,rng,(lo,hi))
        if result is None:return None,'placement_failed'
        occ,patch,donor_ann,meta=result;x,y,p,sm=patch
        pixels[y:y+p.shape[0],x:x+p.shape[1]][sm]=p[sm]
    ratio=float((target&occ).sum()/target.sum())
    if not lo<=ratio<hi:return None,'severity_outside_bin'
    if (target&~occ).sum()<64:return None,'target_visible_too_small'
    newanns=[];changes=[]
    for ann,mask in zip(anns,masks):
        remaining=mask&~occ;removed=int((mask&occ).sum())
        if not removed:newanns.append(copy.deepcopy(ann));continue
        rec=dict(source_ann_id=ann['id'],before_area=int(mask.sum()),after_area=int(remaining.sum()),removed_area=removed)
        changes.append(rec)
        if remaining.sum()==0:continue
        updated=copy.deepcopy(ann);updated['segmentation']=encode(remaining)
        updated['bbox']=mask_utils.toBbox(updated['segmentation']).tolist();updated['area']=int(remaining.sum())
        newanns.append(updated)
    if donor_ann is not None:
        newanns.append(dict(id=-1,image_id=i,category_id=donor_ann['category_id'],segmentation=encode(occ),
                            area=int(occ.sum()),bbox=mask_utils.toBbox(encode(occ)).tolist(),iscrowd=0,ignore=0))
    rec=dict(protocol=VERSION,source_image_id=i,source_file=im['file_name'],target_ann_id=aid,
             target_category_id=cat,family=family,level=level,added_occlusion_ratio=ratio,
             original_visible_target=encode(target),occluder=encode(occ),changes=changes,**meta)
    return (pixels,newanns,rec),None


class DatasetWriter:
    """Stream full-resolution pixels to disk; retain metadata only in RAM."""
    def __init__(self,source,out,name):
        self.source=source;self.out=out;self.name=name
        self.image_dir=out/'images'/name;self.image_dir.mkdir(parents=True,exist_ok=True)
        self.images=[];self.anns=[];self.records=[];self.nextann=1

    def add(self,anchor,result):
        pixels,newanns,rec=result;i=anchor[0]
        rec=copy.deepcopy(rec)
        dest=self.image_dir/(str(i)+'.png')
        if not cv2.imwrite(str(dest),pixels,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError(dest)
        self.images.append(dict(self.source.images[i],file_name=str(dest)))
        for a in newanns:
            sourceid=a['id'];a=dict(a,id=self.nextann,source_ann_id=sourceid);self.nextann+=1
            if sourceid==rec['target_ann_id']:rec['generated_target_ann_id']=a['id']
            self.anns.append(a)
        rec['generated_file']=str(dest);rec['image_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();self.records.append(rec)

    def finish(self):
        data=dict(images=self.images,annotations=self.anns,categories=self.source.categories,info=dict(description=VERSION))
        (self.out/'annotations'/(self.name+'.json')).write_text(json.dumps(data))
        (self.out/'manifests'/(self.name+'.json')).write_text(json.dumps(self.records))
        return self.records


def write_clean(source,ids,out,name):
    (out/'annotations'/(name+'.json')).write_text(json.dumps(source.export(ids)))


def render_preview(rec,out,index):
    before=read_image(rec['source_file']).copy();after=read_image(rec['generated_file']).copy()
    mask=decode(dict(segmentation=rec['original_visible_target']),*before.shape[:2])
    occ=decode(dict(segmentation=rec['occluder']),*before.shape[:2])
    target=after.copy();target[mask&~occ]=(0.4*target[mask&~occ]+0.6*np.array([50,220,50])).astype('uint8')
    target[mask&occ]=(0.4*target[mask&occ]+0.6*np.array([20,30,240])).astype('uint8')
    yy,xx=np.where(mask);h,w=mask.shape
    cx,cy=(xx.min()+xx.max())/2,(yy.min()+yy.max())/2
    cw=max(96.,(xx.max()-xx.min()+1)*2.2);ch=max(96.,(yy.max()-yy.min()+1)*2.2)
    x1,x2=max(0,int(cx-cw/2)),min(w,int(cx+cw/2+1))
    y1,y2=max(0,int(cy-ch/2)),min(h,int(cy+ch/2+1))
    def thumb(im):
        ih,iw=im.shape[:2];s=min(512/iw,288/ih);nw,nh=max(1,round(iw*s)),max(1,round(ih*s))
        canvas=np.full((288,512,3),235,np.uint8)
        canvas[(288-nh)//2:(288-nh)//2+nh,(512-nw)//2:(512-nw)//2+nw]=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA)
        return canvas
    panels=[];zooms=[]
    for im in [before,after,target]:
        zooms.append(thumb(im[y1:y2,x1:x2]))
        overview=im.copy();cv2.rectangle(overview,(x1,y1),(x2,y2),(255,180,0),3);panels.append(thumb(overview))
    header=np.full((46,1536,3),248,np.uint8)
    for k,title in enumerate(['Original','Occluded','Visible target: green / removed: red']):
        cv2.putText(header,title,(k*512+12,29),cv2.FONT_HERSHEY_SIMPLEX,0.60,(25,25,25),1)
    footer=np.full((42,1536,3),248,np.uint8)
    cv2.putText(footer,f"{rec['family']} | {rec['level']} | added visible-mask occlusion={rec['added_occlusion_ratio']:.1%} | bottom row: target zoom",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(25,25,25),1)
    montage=np.concatenate([header,np.concatenate(panels,axis=1),np.concatenate(zooms,axis=1),footer],axis=0)
    path=out/'preview'/f'{index:03d}.jpg';cv2.imwrite(str(path),montage)
    return path.name


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--train-fraction',type=float,default=0.30);ap.add_argument('--test-pairs',type=int,default=300)
    ap.add_argument('--dev-pairs',type=int,default=60);ap.add_argument('--train-limit',type=int,default=0)
    args=ap.parse_args();cv2.setNumThreads(1);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Frozen protocol exists. Use a new output directory for a new version.')
    for sub in ['annotations','manifests','preview','review']: (out/sub).mkdir(exist_ok=True)
    data=Path(args.data)
    src={s:Source(data/f'{s}_seg_json/ins_seg_{s}_coco_bdd8_exists.json',data/s) for s in ['train','val']}
    train,val=src['train'],src['val']
    train_names={Path(x['file_name']).name for x in train.images.values()};val_names={Path(x['file_name']).name for x in val.images.values()}
    if train_names&val_names:raise ValueError('Original train/val image overlap.')
    for source in src.values():
        missing=[v['file_name'] for v in source.images.values() if not Path(v['file_name']).is_file()]
        if missing:raise FileNotFoundError(missing[:5])
    groups=defaultdict(list)
    for i,im in val.images.items():groups[Path(im['file_name']).stem.split('-')[0]].append(i)
    ordered=sorted(groups,key=lambda x:key('val_partition',x));donor_ids=[];dev_ids=[];test_ids=[]
    for g in ordered:
        bucket=donor_ids if len(donor_ids)<100 else dev_ids if len(dev_ids)<198 else test_ids
        bucket.extend(groups[g])
    partitions=dict(donor_val=donor_ids,dev=dev_ids,locked_test=test_ids)
    (out/'partitions.json').write_text(json.dumps(partitions,indent=2))
    write_clean(val,dev_ids,out,'val_dev_clean');write_clean(val,test_ids,out,'val_locked_clean');write_clean(val,list(val.images),out,'val_full_clean')
    audit=dict(protocol=VERSION,train_images=len(train.images),val_images=len(val.images),
               train_annotations=len(train.data['annotations']),val_annotations=len(val.data['annotations']),
               partitions={k:len(v) for k,v in partitions.items()},occluded_attribute_available=any('occluded' in a or 'attributes' in a for a in val.data['annotations']),
               source_sha256={s:hashlib.sha256(x.path.read_bytes()).hexdigest() for s,x in src.items()},
               status='building',rejections={},datasets={})
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    n=args.train_limit or round(len(train.images)*args.train_fraction)
    selected=anchors(train,list(train.images),n);donors=train.candidates(list(train.images))
    pool={p:DatasetWriter(train,out,'train_'+p) for p in ['O1','O2','O3','O4']};rejected=Counter()
    for k,anchor in enumerate(selected):
        level='L' if key('train_level',anchor[1])%2==0 else 'M'
        results={}
        for family in ['geometry','copy']:
            r,e=synthesize(train,anchor,donors,family,level,'train')
            if e:rejected[family+':'+e]+=1
            results[family]=r
        choice='geometry' if key('mixture',anchor[1])%2==0 else 'copy'
        stronglevel=['L','M','M','H'][key('stronglevel',anchor[1])%4]
        strong,e=synthesize(train,anchor,donors,choice,stronglevel,'train_strong')
        if e:rejected['strong:'+e]+=1
        if all(results.values()) and strong is not None:
            for policy,result in [('O1',results['geometry']),('O2',results['copy']),('O3',copy.deepcopy(results[choice])),('O4',strong)]:
                pool[policy].add(anchor,result)
        if (k+1)%100==0:print(json.dumps(dict(stage='train',processed=k+1,accepted=len(pool['O1'].records))),flush=True)
    previews=[]
    for policy,writer in pool.items():
        records=writer.finish();previews.extend(records[:8])
        frac=len(records)/len(train.images);desired=0.25 if policy=='O4' else 0.15
        prob=min(1.0,desired/frac) if frac else 0.0
        audit['datasets'][policy]=dict(candidates=len(records),candidate_fraction=frac,expected_augmented_fraction=frac*prob,
                                       replacement_probability=prob,requested_augmented_fraction=desired,
                                       family_counts=dict(Counter(r['family'] for r in records)),category_counts=dict(Counter(r['target_category_id'] for r in records)))
    del pool
    eval_donors=val.candidates(donor_ids)
    for split,ids,limit in [('dev',dev_ids,args.dev_pairs),('test',test_ids,args.test_pairs)]:
        candidate=anchors(val,ids,len(ids));accepted=[];sets={f'{f}_{l}':DatasetWriter(val,out,f'val_{split}_{f}_{l}') for f in ['geometry','copy'] for l in LEVELS}
        for k,anchor in enumerate(candidate):
            results={};failed=False
            for family in ['geometry','copy']:
                for level in LEVELS:
                    r,e=synthesize(val,anchor,eval_donors,family,level,'val_'+split)
                    if e:rejected[split+':'+family+':'+e]+=1;failed=True;break
                    results[f'{family}_{level}']=r
                if failed:break
            if not failed:
                accepted.append(anchor)
                for name,r in results.items():sets[name].add(anchor,r)
            if len(accepted)>=limit:break
        for name,writer in sets.items():
            records=writer.finish();previews.extend(records[:3])
        write_clean(val,[a[0] for a in accepted],out,f'val_{split}_paired_clean')
        clean_records=[]
        for rec in sets['geometry_L'].records:
            im=val.images[rec['source_image_id']]
            clean_records.append(dict(rec,generated_target_ann_id=rec['target_ann_id'],generated_file=rec['source_file'],
                                      family='clean',level='clean',added_occlusion_ratio=0.,changes=[],
                                      image_sha256=hashlib.sha256(Path(rec['source_file']).read_bytes()).hexdigest(),
                                      occluder=encode(np.zeros((im['height'],im['width']),bool))))
        (out/'manifests'/f'val_{split}_paired_clean.json').write_text(json.dumps(clean_records))
        audit['datasets']['val_'+split]=dict(paired_images=len(accepted),requested=limit,levels=LEVELS,donor_images=len(donor_ids))
    audit['rejections']=dict(rejected)
    review=anchors(val,test_ids,min(300,len(test_ids)))
    with (out/'review/natural_occlusion_review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','ann_id','category_id','file_name','occluded','truncated','reviewer','notes']);writer.writeheader()
        for i,a,c in review:writer.writerow(dict(image_id=i,ann_id=a,category_id=c,file_name=val.images[i]['file_name'],occluded='unreviewed',truncated='unreviewed'))
    cards=[]
    for index,rec in enumerate(previews):
        fn=render_preview(rec,out,index)
        cards.append(f'<figure><img loading="lazy" src="{fn}" width="100%"><figcaption>{html.escape(rec["family"])} / {rec["level"]} / added occlusion {rec["added_occlusion_ratio"]:.1%}</figcaption></figure>')
    (out/'preview/index.html').write_text('<!doctype html><meta charset="utf-8"><title>BDD visible occlusion protocol</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;background:#f5f6f8}figure{background:white;padding:16px}figcaption{margin-top:8px}</style><h1>Original / Occluded / Visible target and removed region</h1><p>Green: remaining visible target. Red: newly hidden original visible pixels. These are construction examples, not predictions.</p>'+''.join(cards))
    audit['status']='complete';(out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'COMPLETE.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
