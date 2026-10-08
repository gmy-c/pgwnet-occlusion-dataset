"""Optional MMDetection dataset branch; original sources stay read-only."""
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from mmdet.registry import HOOKS
from mmengine.hooks import Hook

@DATASETS.register_module()
class LiteBranchDataset(CocoDataset):
    def __init__(self,*args,protocol_root,appearance_probability=.2,occlusion_probability=.1,**kwargs):
        if kwargs.get('test_mode',False):raise ValueError('Training replacement only')
        if not 0<=appearance_probability or not 0<=occlusion_probability or appearance_probability+occlusion_probability>1:raise ValueError('Invalid global mixture')
        super().__init__(*args,**kwargs)
        ids={self.get_data_info(i)['img_id'] for i in range(len(self))};self.alternates={}
        for name in ['appearance','occlusion']:
            ds=CocoDataset(ann_file=str(Path(protocol_root)/'annotations'/f'train_{name}.json'),data_root='',data_prefix=dict(img=''),pipeline=[],metainfo=self.metainfo,serialize_data=False,filter_cfg=dict(filter_empty_gt=False,min_size=1))
            self.alternates[name]={v['img_id']:v for v in ds.data_list if v['img_id'] in ids}
        if set(self.alternates['appearance'])!=ids:raise ValueError('Appearance branch must cover every filtered source')
        k=len(self.alternates['occlusion']);self.p_occ=occlusion_probability*len(self)/k if k else 0
        if (not k and occlusion_probability) or self.p_occ>1:raise ValueError('Not enough eligible occlusion candidates')
        self.p_photo=appearance_probability/(1-occlusion_probability) if occlusion_probability<1 else 0
        self.branch_statistics=dict(source_count=len(self),eligible_occlusion_count=k,p_occ_on_candidates=self.p_occ,p_appearance_after_no_occlusion=self.p_photo,target_global_mix=dict(clean=1-appearance_probability-occlusion_probability,appearance=appearance_probability,occlusion=occlusion_probability))

    def prepare_data(self,idx):
        info=self.get_data_info(idx);identity=info['img_id'];branch='clean'
        if identity in self.alternates['occlusion'] and np.random.random()<self.p_occ:branch='occlusion'
        elif np.random.random()<self.p_photo:branch='appearance'
        if branch!='clean':
            sample_idx=info.get('sample_idx',idx);info=copy.deepcopy(self.alternates[branch][identity]);info['sample_idx']=sample_idx
        info['lite_branch']=branch
        return self.pipeline(info)

@HOOKS.register_module()
class LiteBranchCountHook(Hook):
    def before_train_epoch(self,runner):
        self.counts=Counter()
    def before_train_iter(self,runner,batch_idx,data_batch=None):
        for sample in (data_batch or {}).get('data_samples',[]):
            self.counts[sample.metainfo.get('lite_branch','clean')]+=1
    def after_train_epoch(self,runner):
        if runner.rank==0:
            path=Path(runner.work_dir)/'lite_branch_counts.jsonl'
            with path.open('a') as f:f.write(json.dumps(dict(epoch=runner.epoch+1,counts=dict(self.counts)))+'\n')
