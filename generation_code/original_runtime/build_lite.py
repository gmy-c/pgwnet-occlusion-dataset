"""Build isolated full-resolution PNG datasets and exact visible-mask labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
import argparse,copy,hashlib,importlib.util,json,os,shutil,time,traceback
import cv2
import numpy as np
from pycocotools import mask as mu
from lite_operators import VERSION,FAMILIES,LEVELS,seed_for,apply_rgb

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_png(path,rgb):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.tmp.png')
    assert cv2.imwrite(str(tmp),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1])
    tmp.replace(path)
    check=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    assert np.array_equal(check,rgb),'PNG roundtrip mismatch'
def get_rgb(path):
    im=cv2.imread(str(path))
    if im is None:raise IOError(path)
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--legacy-root',required=True);ap.add_argument('--data',required=True);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--pilot',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    cv2.setNumThreads(1);out=Path(a.out).resolve();legacy=Path(a.legacy_root);data=Path(a.data)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():raise RuntimeError('Frozen build exists; refuse overwrite')
    spec=importlib.util.spec_from_file_location('legacy_builder',legacy/'tools/build_protocol.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    train=old.Source(data/'train_seg_json/ins_seg_train_coco_bdd8_exists.json',data/'train')
    val=old.Source(data/'val_seg_json/ins_seg_val_coco_bdd8_exists.json',data/'val')
    assert len(train.images)==6996 and len(val.images)==998
    assert train.categories==val.categories
    assert not ({Path(v['file_name']).name for v in train.images.values()} & {Path(v['file_name']).name for v in val.images.values()})
    begin=time.time();state={'version':VERSION,'status':'building','started':begin,'out':str(out),'workers':a.workers}
    def progress(stage,**kw):
        state.update(stage=stage,updated=time.time(),**kw);dump(out/'build_state.json',state);print(json.dumps(state),flush=True)
    sources={};checked=0
    for split,src in [('train',train),('val',val)]:
        for i,im in src.images.items():
            p=Path(im['file_name']);assert p.is_file(); rgb=get_rgb(p)
            assert rgb.shape[:2]==(im['height'],im['width']);checked+=1
            sources[split+':'+str(i)]={'image_id':i,'file_name':str(p),'sha256':sha(p),'width':im['width'],'height':im['height']}
            if checked%500==0:progress('source_audit',source_images_checked=checked)
        dump(out/'annotations'/f'{split}_clean.json',src.export(list(src.images)))
    dump(out/'source_manifest.json',{'images':sources,'annotation_sha256':{s:sha(x.path) for s,x in [('train',train),('val',val)]},'categories':train.categories})
    if shutil.disk_usage(out).free<50*1024**3 and not a.pilot:raise RuntimeError('Require 50 GiB free before full generation')
    train_ids=sorted(train.images,key=lambda i:seed_for('allocation',i))
    plans=defaultdict(list)
    for j,i in enumerate(train_ids[:50] if a.pilot else train_ids):
        family=FAMILIES[j%4];level=LEVELS[(j//4)%2]
        plans['train_appearance'].append((i,family,level,'train'))
    for family in FAMILIES:
        for level in LEVELS:
            plans[f'val_{family}_{level}']=[(i,family,level,'val') for i in (list(val.images)[:10] if a.pilot else val.images)]
    counts={};all_rec=[];first_bytes=[]
    for name,items in plans.items():
        src=train if name.startswith('train') else val
        cached={}
        saved=out/'manifests'/f'{name}.json'
        if a.resume and saved.exists():
            cached={r['source_image_id']:r for r in json.loads(saved.read_text())}
        def process(task):
            i,fam,lev,split=task;im=src.images[i];rngseed=seed_for(split,i,fam,'appearance');path=out/'images'/name/f'{i}.png'
            rec=cached.get(i)
            if rec and rec['family']==fam and rec['level']==lev and rec['seed']==rngseed and rec['source_file']==im['file_name'] and rec['source_sha256']==sources[split+':'+str(i)]['sha256'] and rec['generated_file']==str(path) and path.is_file() and sha(path)==rec['sha256']:
                return rec
            image=get_rgb(im['file_name']);rendered,params=apply_rgb(image,fam,lev,rngseed)
            write_png(path,rendered)
            return {'view_id':name+':'+str(i),'source_image_id':i,'source_split':split,'source_file':im['file_name'],'source_sha256':sources[split+':'+str(i)]['sha256'],'family':fam,'level':lev,'seed':rngseed,'parameters':params,'generated_file':str(path),'sha256':sha(path),'label_policy':'original_geometry_inherited','qa':'png_exact_roundtrip_passed','bytes':path.stat().st_size}
        records=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for rec in pool.map(process,items):
                records.append(rec)
                if len(records)%100==0:progress('appearance',dataset=name,done=len(records),total=len(items))
        annotation=copy.deepcopy(src.export([r['source_image_id'] for r in records]));paths={r['source_image_id']:r['generated_file'] for r in records}
        for im in annotation['images']:im['file_name']=paths[im['id']]
        dump(out/'annotations'/f'{name}.json',annotation);dump(out/'manifests'/f'{name}.json',records)
        counts[name]=len(records);all_rec+=records;progress('appearance_completed',dataset=name,count=len(records))
    if a.pilot:
        dump(out/'PILOT_COMPLETE.json',{'counts':counts,'seconds':time.time()-begin,'mean_png_bytes':sum(r['bytes'] for r in all_rec)/len(all_rec)})
        progress('pilot_complete',status='pilot_complete');return
    # Reuse legacy train copy-paste pixels, but verify every derived mask against source truth.
    old_ann=json.loads((legacy/'protocol/annotations/train_O2.json').read_text());old_rec=json.loads((legacy/'protocol/manifests/train_O2.json').read_text())
    old_byid=defaultdict(list)
    for ann in old_ann['annotations']:old_byid[ann['image_id']].append(ann)
    train_records=[];verified=0;canonicalized=0;empty_generated_ids=set()
    for k,rec in enumerate(old_rec):
        i=rec['source_image_id'];im=train.images[i];occ=old.decode({'segmentation':rec['occluder']},im['height'],im['width']); generated=old_byid[i]
        mapping={x['source_ann_id']:x for x in generated if x.get('source_ann_id',-1)!=-1};empty_source_ids=[]
        for ann,mask in zip(train.anns[i],train.masks(i)):
            expected=mask&~occ
            if expected.any():
                assert ann['id'] in mapping
                z=mapping[ann['id']];assert np.array_equal(old.decode(z,im['height'],im['width']),expected)
                # Unchanged legacy polygons retain analytic area; canonicalize every mask.
                encoded=old.encode(expected)
                canonicalized+=int(z['area']!=int(expected.sum()) or not np.allclose(z['bbox'],mu.toBbox(encoded)))
                z['segmentation']=encoded;z['area']=int(expected.sum());z['bbox']=mu.toBbox(encoded).tolist()
            elif ann['id'] in mapping:
                # Subpixel legacy polygons can rasterize to an empty mask before occlusion.
                assert not mask.any(),'Legacy unexpectedly retained a fully occluded positive-area mask'
                empty_generated_ids.add(mapping[ann['id']]['id']);empty_source_ids.append(ann['id'])
            verified+=1
        donor=[x for x in generated if x.get('source_ann_id')==-1];assert len(donor)==1
        assert np.array_equal(old.decode(donor[0],im['height'],im['width']),occ)
        path=out/'images/train_occlusion'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(rec['generated_file'],path);assert sha(path)==rec['image_sha256']
        train_records.append(dict(rec,generated_file=str(path),label_policy='verified_legacy_train_copy_subtraction_and_rle_canonicalization',raster_empty_source_ann_ids=empty_source_ids,sha256=sha(path)))
        if (k+1)%100==0:progress('train_occlusion_audit',done=k+1,total=len(old_rec))
    paths={r['source_image_id']:r['generated_file'] for r in train_records}
    for im in old_ann['images']:im['file_name']=paths[im['id']]
    old_ann['annotations']=[v for v in old_ann['annotations'] if v['id'] not in empty_generated_ids]
    old_ann['info']={'description':VERSION,'legacy_source':'bdd-visible-occ-v1.0','mask_policy':'all nonempty visible masks canonicalized to RLE with raster area and bbox'}
    dump(out/'annotations/train_occlusion.json',old_ann);dump(out/'manifests/train_occlusion.json',train_records);counts['train_occlusion']=len(train_records)
    # Fixed donor identity and monotone entry trajectory per evaluation source.
    parts=json.loads((legacy/'protocol/partitions.json').read_text());donor_ids=sorted(parts['donor_val'],key=lambda i:seed_for('donor_partition',i))
    donor_pools={'dev':donor_ids[:50],'report':donor_ids[50:]};dump(out/'partitions.json',{'legacy_source_partitions':parts,'donor_pools':donor_pools,'claim':'development_validation_not_blind_test'})
    reject=Counter();occsets={};total_occ_checks=0
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
                    if bounds[0]<=ratio<bounds[1] and target_area*(1-ratio)>=64:
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
    # Two fixed medium compound conditions, identical occlusion labels.
    for family in ['low_exposure_noise','motion_blur']:
        name='val_compound_'+family+'_occlusion_M';records=[];ann=json.loads((out/'annotations/val_occlusion_report_M.json').read_text())
        for rec in occsets[('report','M')]:
            i=rec['source_image_id'];path=out/'images'/name/f'{i}.png';seed=seed_for('val',i,family,'appearance');rgb,params=apply_rgb(get_rgb(rec['generated_file']),family,'M',seed);write_png(path,rgb)
            records.append(dict(rec,generated_file=str(path),family=name,parameters=params,seed=seed,sha256=sha(path)))
        paths={r['source_image_id']:r['generated_file'] for r in records}
        for im in ann['images']:im['file_name']=paths[im['id']]
        dump(out/'annotations'/f'{name}.json',ann);dump(out/'manifests'/f'{name}.json',records);counts[name]=len(records)
    # Fixed small development panel and manifest-backed evaluation suites.
    panel=sorted(parts['dev'],key=lambda i:seed_for('panel',i))[:100]
    for family in FAMILIES:
        full=json.loads((out/'annotations'/f'val_{family}_M.json').read_text());sub=dict(full,images=[v for v in full['images'] if v['id'] in panel],annotations=[v for v in full['annotations'] if v['image_id'] in panel]);dump(out/'annotations'/f'dev_{family}_M.json',sub)
    clean=['val_clean'];appearance=[f'val_{f}_{l}' for f in FAMILIES for l in LEVELS];occ=[f'val_occlusion_report_{l}' for l in LEVELS];comp=[f'val_compound_{f}_occlusion_M' for f in ['low_exposure_noise','motion_blur']]
    suites={'clean':clean,'lite':appearance+occ+comp,'all':clean+appearance+occ+comp,'dev':[f'dev_{f}_M' for f in FAMILIES]+['val_occlusion_dev_M'],'paired_clean':['val_occlusion_report_clean']}
    dump(out/'eval_suites.json',suites);dump(out/'rejections.json',dict(reject))
    # Compact previews, never model predictions.
    preview=[]
    for fam in FAMILIES:
        i=list(val.images)[0];tiles=[get_rgb(val.images[i]['file_name'])]+[get_rgb(out/'images'/f'val_{fam}_{l}'/f'{i}.png') for l in LEVELS]
        row=np.concatenate([cv2.resize(v,(384,216)) for v in tiles],axis=1);path=out/'preview'/f'{fam}.png';write_png(path,row);preview.append(str(path))
    for k,rec in enumerate(occsets[('report','M')][:6]):old.render_preview(rec,out,k)
    count_main=998+sum(counts[n] for n in appearance+occ);assert count_main==13874
    receipt={'protocol':VERSION,'status':'complete','counts':counts,'main_eval_views_including_clean':count_main,'full_eval_with_compound':count_main+600,'train_pool_views':6996+counts['train_appearance']+counts['train_occlusion'],'extra_occlusion_dev_views':180,'source_images':{'train':6996,'val':998},'new_manual_annotations':0,'train_occlusion_annotation_checks':verified,'new_occlusion_annotation_checks':total_occ_checks,'source_annotation_sha256':{s:sha(x.path) for s,x in [('train',train),('val',val)]},'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'seconds':time.time()-begin,'rejections':dict(reject),'split_isolation':True,'paired_donor_identity':True,'storage':'lossless PNG materialized on shared filesystem','previews':preview,'eval_claim':'development validation; not blind test','compatibility_smoke':'pending'}
    receipt['canonicalized_legacy_annotation_geometry']=canonicalized
    receipt['removed_legacy_raster_empty_annotations']=len(empty_generated_ids)
    dump(out/'BUILD_COMPLETE.json',receipt);progress('built_waiting_final_validation',status='built',counts=counts)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
