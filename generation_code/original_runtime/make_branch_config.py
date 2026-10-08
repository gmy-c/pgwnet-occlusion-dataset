"""Compose an existing model recipe with independent training/evaluation choices."""
import argparse,copy,json,hashlib
from pathlib import Path
from mmengine.config import Config

def bitmap(pipeline):
    for step in pipeline:
        if step.get('type')=='LoadAnnotations':step['poly2mask']=True
        if step.get('type')=='PackDetInputs':
            keys=tuple(step.get('meta_keys',('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction')))
            if 'lite_branch' not in keys:step['meta_keys']=keys+('lite_branch',)
def compose(base,root,mode,work_dir):
    c=Config.fromfile(base);root=Path(root)
    base_digest=hashlib.sha256(c.pretty_text.encode()).hexdigest()
    protected={k:copy.deepcopy(c[k]) for k in ['model','load_from','optim_wrapper','param_scheduler','randomness'] if k in c}
    imports=c.get('custom_imports',dict(imports=[],allow_failed_imports=False))
    imports['imports']=list(imports['imports'])+['lite_dataset'];c.custom_imports=imports
    tr=c.train_dataloader.dataset
    if tr.get('type') not in ['CocoDataset','PairedOcclusionDataset','LiteBranchDataset']:raise ValueError('Unsupported wrapped dataset; explicit adaptation required')
    for key in ['augmented_ann_file','replace_probability','protocol_root','appearance_probability','occlusion_probability']:tr.pop(key,None)
    tr.type='LiteBranchDataset' if mode=='lite' else 'CocoDataset';tr.ann_file=str(root/'annotations/train_clean.json');tr.data_root='';tr.data_prefix=dict(img='')
    if mode=='lite':tr.update(protocol_root=str(root),appearance_probability=.2,occlusion_probability=.1)
    bitmap(tr.pipeline)
    for hook in c.get('custom_hooks',[]):
        if hook.get('type')=='PipelineSwitchHook':bitmap(hook['switch_pipeline'])
    c.custom_hooks=list(c.get('custom_hooks',[]))+[dict(type='LiteBranchCountHook')]
    # Both newly trained controls use bitmap masks and the SAME full clean validation.
    c.val_cfg=dict(type='ValLoop');c.val_dataloader.dataset.type='CocoDataset';c.val_dataloader.dataset.ann_file=str(root/'annotations/val_clean.json');c.val_dataloader.dataset.data_root='';c.val_dataloader.dataset.data_prefix=dict(img='');bitmap(c.val_dataloader.dataset.pipeline)
    c.val_evaluator.ann_file=str(root/'annotations/val_clean.json');c.default_hooks.checkpoint.save_best='coco/segm_mAP';c.train_cfg.val_interval=1
    c.test_dataloader=copy.deepcopy(c.val_dataloader);c.test_evaluator=copy.deepcopy(c.val_evaluator);c.test_cfg=dict(type='TestLoop')
    c.work_dir=str(work_dir);c.resume=False;c.lite_protocol=dict(root=str(root),train_mode=mode,selection='clean full998 coco/segm_mAP',bitmap_control=True,source_base=str(Path(base).resolve()))
    if 'experiment' in c:
        c.experiment.policy=mode;c.experiment.primary_selection='coco/segm_mAP';c.experiment.dataset_protocol='bdd-lite-v1.0'
    assert all(c[k]==v for k,v in protected.items()),'Model/initialization/training recipe drift'
    c.lite_protocol.base_resolved_sha256=base_digest;c.lite_protocol.preserved_fields=list(protected)
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--protocol-root',required=True);p.add_argument('--train-mode',choices=['clean','lite'],required=True);p.add_argument('--work-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.protocol_root)
    if not (root/'COMPLETE.json').exists():raise RuntimeError('Protocol must pass final validation before exporting training config')
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);compose(a.base_config,root,a.train_mode,a.work_dir).dump(str(path));print(path)
if __name__=='__main__':main()
