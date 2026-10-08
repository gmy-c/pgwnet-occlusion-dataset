"""Validate frozen artifacts and exercise dataset switches with real samples on CPU."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time,platform,importlib.metadata
from unittest.mock import patch
import cv2,numpy as np,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol-root',required=True);p.add_argument('--model-root',required=True);p.add_argument('--check-all-png',action='store_true');a=p.parse_args()
    root=Path(a.protocol_root);modelroot=Path(a.model_root);torch.set_num_threads(1);cv2.setNumThreads(1)
    sys.path[:0]=[str(modelroot),str(modelroot/'project'),str(Path(__file__).parent)]
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from mmengine.utils import import_modules_from_strings
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS,MODELS
    from make_branch_config import compose
    import lite_dataset
    receipt=json.loads((root/'BUILD_COMPLETE.json').read_text());assert receipt['status']=='complete'
    assert receipt['train_pool_views']==15909 and receipt['main_eval_views_including_clean']==13874
    suites=json.loads((root/'eval_suites.json').read_text());checked=0
    for name in set(n for v in suites.values() for n in v)|{'train_clean','train_appearance','train_occlusion'}:
        ds=json.loads((root/'annotations'/f'{name}.json').read_text());ids={v['id'] for v in ds['images']};assert len(ids)==len(ds['images']);annids={v['id'] for v in ds['annotations']};assert len(annids)==len(ds['annotations'])
        assert all(v['image_id'] in ids for v in ds['annotations'])
        for im in ds['images']:
            assert Path(im['file_name']).is_file()
            if a.check_all_png:
                pixels=cv2.imread(im['file_name']);assert pixels is not None and pixels.shape[:2]==(im['height'],im['width']);checked+=1
    # Independent cross-level consistency checks on all paired validation records.
    for split,n in [('dev',60),('report',300)]:
        rows=[json.loads((root/'manifests'/f'val_occlusion_{split}_{lev}.json').read_text()) for lev in ['L','M','H']]
        assert all(len(v)==n for v in rows)
        for triple in zip(*rows):
            assert len({(v['source_image_id'],v['target_ann_id'],v['donor_image_id'],v['donor_ann_id'],v['scale'],v['y']) for v in triple})==1
            assert triple[0]['added_occlusion_ratio']<triple[1]['added_occlusion_ratio']<triple[2]['added_occlusion_ratio']
    from audit_annotations import audit
    annotation_audit=audit(root,modelroot)
    summary={'status':'passed','validated_at':time.time(),'image_reads':checked,'counts':receipt['counts'],'modes':{},'paired_validation_sources':360,'annotation_audit':annotation_audit}
    configs=root/'configs';configs.mkdir(exist_ok=True)
    for mode in ['clean','lite']:
        c=compose(str(modelroot/'configs/F1_clean.py'),root,mode,root/'runs'/('EXAMPLE_F1_'+mode));import_modules_from_strings(**c.custom_imports)
        ds=DATASETS.build(c.train_dataloader.dataset);summary['modes'][mode]={'length':len(ds),'dataset_type':type(ds).__name__};c.dump(str(configs/f'example_F1_{mode}.py'))
        (root/f'filtered_train_ids_{mode}.json').write_text(json.dumps([ds.get_data_info(j)['img_id'] for j in range(len(ds))]))
        if mode=='lite':
            assert len(ds)==summary['modes']['clean']['length'];summary['modes'][mode]['sampling']=ds.branch_statistics
            j=next(j for j in range(len(ds)) if ds.get_data_info(j)['img_id'] in ds.alternates['occlusion'])
            samples=[]
            for branch,rand in [('clean',.999999),('appearance',0.0),('occlusion',0.0)]:
                original=ds.p_occ
                if branch=='appearance':ds.p_occ=0
                with patch('numpy.random.random',return_value=rand):sample=ds.prepare_data(j)
                ds.p_occ=original;assert sample is not None
                assert sample['data_samples'].metainfo['lite_branch']==branch
                assert type(sample['data_samples'].gt_instances.masks).__name__=='BitmapMasks';samples.append(sample)
            summary['modes'][mode]['branches_exercised']=['clean','appearance','occlusion']
            # Run a real CPU loss/backward using reduced spatial size; no formal training/weights saved.
            smoke=copy.deepcopy(c.train_dataloader.dataset)
            for step in smoke.pipeline:
                if step['type']=='Resize':step['scale']=(320,192)
            small=DATASETS.build(smoke)
            with patch('numpy.random.random',return_value=0.0):batch=pseudo_collate([small.prepare_data(j)])
            model=MODELS.build(c.model).cpu();model.train()
            data=model.data_preprocessor(batch,training=True);losses=model(**data,mode='loss');terms=[]
            for key,value in losses.items():
                if 'loss' in key:terms.extend(value if isinstance(value,list) else [value])
            loss=sum(v.mean() for v in terms);assert torch.isfinite(loss);loss.backward()
            summary['cpu_loss_backward']={'passed':True,'loss':float(loss.detach()),'input_shape':list(data['inputs'].shape),'annotation_branch':'occlusion','formal_training':False}
            model.eval()
            with torch.no_grad():prediction=model.predict(data['inputs'],data['data_samples'],rescale=False)
            assert len(prediction)==1 and 'pred_instances' in prediction[0]
            summary['cpu_predict']={'passed':True,'instances':len(prediction[0].pred_instances),'rescale':False,'formal_AP':False}
            # The E2 stage-2 crop/resize/HSV path must also accept RLE masks.
            from mmcv.transforms import Compose
            stage2=[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True,with_mask=True,poly2mask=True),dict(type='RandomResize',scale=(320,192),ratio_range=(.5,1.5),keep_ratio=True),dict(type='RandomCrop',crop_size=(160,96),allow_negative_crop=True,recompute_bbox=True),dict(type='FilterAnnotations',min_gt_bbox_wh=(1,1)),dict(type='YOLOXHSVRandomAug'),dict(type='RandomFlip',prob=.5),dict(type='Pad',size=(160,96),pad_val=dict(img=(114,114,114))),dict(type='PackDetInputs')]
            example=copy.deepcopy(ds.alternates['occlusion'][ds.get_data_info(j)['img_id']]);packed=Compose(stage2)(example)
            assert packed is not None and type(packed['data_samples'].gt_instances.masks).__name__=='BitmapMasks'
            summary['stage2_bitmap_pipeline']='passed at reduced scale'
    # Reproducibility of each appearance operator on actual image.
    from lite_operators import apply_rgb,FAMILIES
    source=json.loads((root/'annotations/val_clean.json').read_text())['images'][0]['file_name'];image=cv2.cvtColor(cv2.imread(source),cv2.COLOR_BGR2RGB)
    for family in FAMILIES:
        x,_=apply_rgb(image,family,'M',17);y,_=apply_rgb(image,family,'M',17);assert np.array_equal(x,y)
    summary['deterministic_operators']=True
    summary['environment']={'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,'torch':torch.__version__}
    for name in ['mmengine','mmcv','mmdet','pycocotools']:
        try:summary['environment'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:summary['environment'][name]='source installation'
    receipt['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    receipt['legacy_builder_sha256']=hashlib.sha256((modelroot/'tools/build_protocol.py').read_bytes()).hexdigest()
    (root/'VALIDATION.json').write_text(json.dumps(summary,indent=2));receipt['compatibility_smoke']='passed';receipt['status']='complete';receipt['validation']=str(root/'VALIDATION.json');(root/'COMPLETE.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
