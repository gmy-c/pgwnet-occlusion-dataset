"""Independent verification of source isolation, inherited labels and visible masks."""
from pathlib import Path
from collections import Counter,defaultdict
import importlib.util,json,hashlib
import numpy as np
import cv2
from pycocotools import mask as mu

def audit(root,legacy):
    spec=importlib.util.spec_from_file_location('audit_legacy',Path(legacy)/'tools/build_protocol.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    src={s:old.Source(root/'annotations'/f'{s}_clean.json',Path('/')) for s in ['train','val']}
    parts=json.loads((root/'partitions.json').read_text());p=parts['legacy_source_partitions'];donors=parts['donor_pools']
    pools=[set(p[x]) for x in ['donor_val','dev','locked_test']]
    assert not(pools[0]&pools[1] or pools[0]&pools[2] or pools[1]&pools[2]);assert set.union(*pools)==set(src['val'].images)
    assert not(set(donors['dev'])&set(donors['report']));assert set(donors['dev']+donors['report'])==pools[0]
    summary={'split_isolation':'passed','inherited_annotations':0,'occlusion_original_masks_checked':0,'donor_masks_checked':0,'occlusion_pixel_compositions_checked':0,'source_masks_decoded':0,'source_empty_masks':0,'sets':{}}
    from lite_operators import apply_rgb,FAMILIES,seed_for
    summary['appearance_source_links_checked']=0;summary['appearance_reproductions']=0
    for manifest in sorted((root/'manifests').glob('*.json')):
        name=manifest.stem
        if name!='train_appearance' and not any(name.startswith('val_'+f+'_') for f in FAMILIES):continue
        records=json.loads(manifest.read_text());split='train' if name.startswith('train') else 'val';s=src[split]
        selected=set(np.linspace(0,len(records)-1,min(16,len(records))).astype(int).tolist())
        for j,rec in enumerate(records):
            source=s.images[rec['source_image_id']]['file_name'];assert rec['source_file']==source
            assert rec['seed']==seed_for(split,rec['source_image_id'],rec['family'],'appearance');summary['appearance_source_links_checked']+=1
            if j in selected:
                image=cv2.cvtColor(cv2.imread(source),cv2.COLOR_BGR2RGB);expected,params=apply_rgb(image,rec['family'],rec['level'],rec['seed'])
                actual=cv2.cvtColor(cv2.imread(rec['generated_file']),cv2.COLOR_BGR2RGB);assert np.array_equal(expected,actual);assert params==rec['parameters'];summary['appearance_reproductions']+=1
    for split,s in src.items():
        for i in s.images:
            for mask in s.masks(i):
                summary['source_masks_decoded']+=1;summary['source_empty_masks']+=int(not mask.any())
    for path in sorted((root/'annotations').glob('*.json')):
        name=path.stem;data=json.loads(path.read_text());split='train' if name.startswith('train') else 'val';s=src[split]
        assert data['categories']==s.categories
        ids={v['id'] for v in data['images']};byid=defaultdict(list)
        for ann in data['annotations']:byid[ann['image_id']].append(ann)
        counts=Counter(a['category_id'] for a in data['annotations'])
        summary['sets'][name]={'images':len(ids),'instances':len(data['annotations']),'category_instances':dict(counts)}
        manifest=root/'manifests'/f'{name}.json'
        isocc=(name=='train_occlusion' or name.startswith('val_occlusion_')) and not name.endswith('_clean')
        if name.startswith('val_compound_'):
            ref=json.loads((root/'annotations/val_occlusion_report_M.json').read_text())
            assert data['annotations']==ref['annotations'];summary['inherited_annotations']+=len(data['annotations']);continue
        if not isocc:
            for i in ids:
                assert byid[i]==s.anns[i];summary['inherited_annotations']+=len(byid[i])
            continue
        records=json.loads(manifest.read_text());assert len(records)==len(ids)
        target_classes=Counter();ratios=[];removed=0
        for rec in records:
            i=rec['source_image_id'];assert i in ids and Path(rec['source_file']).resolve()==Path(s.images[i]['file_name']).resolve()
            di=rec['donor_image_id'];da=rec['donor_ann_id'];assert di in s.images
            donor=next(a for a in s.anns[di] if a['id']==da)
            if split=='val':
                pool='dev' if '_dev_' in name else 'report';assert di in donors[pool];assert i in p['dev' if pool=='dev' else 'locked_test']
            h,w=s.images[i]['height'],s.images[i]['width'];occ=old.decode(dict(segmentation=rec['occluder']),h,w)
            # Verify pixels independently, including actual donor identity and untouched background.
            original=cv2.imread(s.images[i]['file_name']);generated=cv2.imread(rec['generated_file'])
            assert np.array_equal(original[~occ],generated[~occ])
            dm=old.decode(donor,s.images[di]['height'],s.images[di]['width']);yy,xx=np.where(dm)
            donor_image=cv2.imread(s.images[di]['file_name']);cut=dm[yy.min():yy.max()+1,xx.min():xx.max()+1];crop=donor_image[yy.min():yy.max()+1,xx.min():xx.max()+1]
            dh=max(2,round(cut.shape[0]*rec['scale']));dw=max(2,round(cut.shape[1]*rec['scale']));small=cv2.resize(cut.astype(np.uint8),(dw,dh),interpolation=cv2.INTER_NEAREST).astype(bool);pixels=cv2.resize(crop,(dw,dh),interpolation=cv2.INTER_LINEAR)
            x,y=rec['x'],rec['y'];reconstructed=np.zeros_like(occ);reconstructed[y:y+dh,x:x+dw]=small
            assert np.array_equal(reconstructed,occ);assert np.array_equal(generated[y:y+dh,x:x+dw][small],pixels[small]);summary['occlusion_pixel_compositions_checked']+=1
            mapping={a['source_ann_id']:a for a in byid[i]};assert len(mapping)==len(byid[i]);assert -1 in mapping
            newdonor=mapping[-1];assert newdonor['category_id']==donor['category_id'];assert np.array_equal(old.decode(newdonor,h,w),occ)
            assert int(newdonor['area'])==int(occ.sum());assert np.allclose(newdonor['bbox'],mu.toBbox(old.encode(occ)));summary['donor_masks_checked']+=1
            for ann,mask in zip(s.anns[i],s.masks(i)):
                visible=mask&~occ;summary['occlusion_original_masks_checked']+=1
                if not visible.any():assert ann['id'] not in mapping;removed+=1;continue
                actual=mapping[ann['id']];assert actual['category_id']==ann['category_id'];assert np.array_equal(old.decode(actual,h,w),visible)
                assert int(actual['area'])==int(visible.sum());assert np.allclose(actual['bbox'],mu.toBbox(old.encode(visible)))
                if ann['id']==rec['target_ann_id']:
                    ratio=float((mask&occ).sum()/mask.sum());assert abs(ratio-rec['added_occlusion_ratio'])<1e-10
                    lo,hi=old.LEVELS[rec['level']];assert lo<=ratio<hi and visible.sum()>=64;target_classes[ann['category_id']]+=1;ratios.append(ratio)
            assert hashlib.sha256(Path(rec['generated_file']).read_bytes()).hexdigest()==rec.get('sha256',rec.get('image_sha256'))
        summary['sets'][name].update(target_classes=dict(target_classes),ratio_min=min(ratios),ratio_max=max(ratios),fully_removed_original_instances=removed)
    # Recover exact attempted candidates from deterministic ordering and accepted anchors.
    rejections=[];split_coverage={};reported=json.loads((root/'rejections.json').read_text())
    for split,key in [('dev','dev'),('report','locked_test')]:
        accepted=json.loads((root/'manifests'/f'val_occlusion_{split}_M.json').read_text());accepted_keys={(v['source_image_id'],v['target_ann_id']) for v in accepted}
        ordered=old.anchors(src['val'],p[key],len(p[key]));last=(accepted[-1]['source_image_id'],accepted[-1]['target_ann_id']);rejected=0
        for i,aid,category in ordered:
            if (i,aid) not in accepted_keys:
                ann=next(v for v in src['val'].anns[i] if v['id']==aid);im=src['val'].images[i];area=int(old.decode(ann,im['height'],im['width']).sum())
                rejections.append(dict(split=split,source_image_id=i,target_ann_id=aid,category_id=category,raster_area=area,reason='raster_target_below_128' if area<128 else 'no_shared_donor_trajectory_within_100_attempts'));rejected+=1
            if (i,aid)==last:break
        assert rejected==reported.get(split+':no_shared_donor_trajectory',0)
        split_coverage[split]=dict(source_pool=len(p[key]),eligible_unique_sources=len(ordered),accepted=len(accepted),rejected_attempted=rejected,eligible_unattempted=len(ordered)-len(accepted)-rejected)
    summary['occlusion_source_coverage']=split_coverage
    (root/'rejected_candidates.json').write_text(json.dumps(rejections,indent=2))
    (root/'ANNOTATION_AUDIT.json').write_text(json.dumps(summary,indent=2));return summary
