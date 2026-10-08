"""Evaluate ANY compatible checkpoint on clean/lite/all without retraining."""
import argparse,csv,hashlib,json,os,re,subprocess,sys
from pathlib import Path
from mmengine.config import Config

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--protocol-root',required=True);p.add_argument('--project',required=True);p.add_argument('--eval-mode',choices=['clean','lite','all','dev','paired_clean'],default='all');p.add_argument('--out',required=True);p.add_argument('--only',nargs='*');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    root=Path(a.protocol_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert (root/'COMPLETE.json').exists()
    suites=json.loads((root/'eval_suites.json').read_text());names=a.only or suites[a.eval_mode]
    allowed={n for values in suites.values() for n in values};assert set(names)<=allowed
    digest=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();base=Config.fromfile(a.base_config)
    binding={'checkpoint':str(Path(a.checkpoint).resolve()),'sha256':digest,'base_config':str(Path(a.base_config).resolve()),'resolved_config_sha256':hashlib.sha256(base.pretty_text.encode()).hexdigest(),'protocol_receipt_sha256':hashlib.sha256((root/'COMPLETE.json').read_bytes()).hexdigest(),'eval_mode':a.eval_mode,'conditions':names}
    binding_path=out/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:raise RuntimeError('Output already bound to different checkpoint/config/suite')
    binding_path.write_text(json.dumps(binding,indent=2));results={}
    for name in names:
        cfg=Config.fromfile(a.base_config);ann=root/'annotations'/f'{name}.json';assert ann.exists()
        cfg.test_dataloader.dataset.type='CocoDataset';cfg.test_dataloader.dataset.ann_file=str(ann);cfg.test_dataloader.dataset.data_root='';cfg.test_dataloader.dataset.data_prefix=dict(img='');cfg.test_dataloader.num_workers=0;cfg.test_dataloader.persistent_workers=False
        for step in cfg.test_dataloader.dataset.pipeline:
            if step.get('type')=='LoadAnnotations':step['poly2mask']=True
        cfg.test_evaluator.ann_file=str(ann);cfg.work_dir=str(out/name);cfg.test_evaluator.outfile_prefix=str(out/name/'predictions');path=out/(name+'.py');cfg.dump(str(path))
        if a.prepare_only:continue
        done=out/(name+'.metrics.json')
        if done.exists():results[name]=json.loads(done.read_text());continue
        log=out/(name+'.log')
        with log.open('w') as f:rc=subprocess.run([sys.executable,str(Path(a.project)/'tools/test.py'),str(path),a.checkpoint],cwd=a.project,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError(str(log))
        rows=[v for v in log.read_text().splitlines() if 'Epoch(test)' in v and 'coco/segm_mAP:' in v];assert rows
        metrics={k:float(v) for k,v in re.findall(r'(coco/\w+):\s*([-+\d.eE]+)',rows[-1])}
        results[name]={'images':len(json.loads(ann.read_text())['images']),'metrics':metrics,'checkpoint_sha256':digest};done.write_text(json.dumps(results[name],indent=2));(out/'summary.json').write_text(json.dumps(results,indent=2))
    if not a.prepare_only:
        (out/'COMPLETED.json').write_text(json.dumps(binding,indent=2))
        keys=sorted({k for row in results.values() for k in row['metrics']})
        with (out/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['condition','images']+keys);w.writeheader()
            for name,row in results.items():w.writerow(dict(condition=name,images=row['images'],**row['metrics']))
    print(json.dumps({'prepared':len(names),'evaluated':len(results),'out':str(out)}))
if __name__=='__main__':main()
