"""Recompute Result 4 from the authors' public deposit, with explicit split units.

No parameter search is performed. Ridge alpha remains 1.0 throughout.
Original scores are regenerated before corrections are interpreted.
Usage: python -m analysis.result4 --stage main --output outputs/result4
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from threadpoolctl import threadpool_limits


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def scale_pair(train, test):
    # Preserve the original dtype and 1e-12 cutoff; change only fit scope.
    mean, std = train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (train-mean)/std, (test-mean)/std


def evaluate(X, y, splits, scaling='train'):
    """Pooled out-of-fold score, fold scores, and predictions. Folds may overlap."""
    if scaling not in {'global', 'train', 'none'}:
        raise ValueError('scaling must be global, train, or none')
    if scaling == 'global':
        X, _ = scale_pair(X, X)
    folds, ys, ps, indices = [], [], [], []
    for fold, (tr, te) in enumerate(splits):
        tr, te = np.asarray(tr), np.asarray(te)
        if not len(tr) or not len(te) or np.intersect1d(tr, te).size:
            raise ValueError('Training and test indices must be nonempty and disjoint')
        if not set(np.unique(y[te])).issubset(set(np.unique(y[tr]))):
            raise ValueError('Closed-set classification requires all test labels in training')
        a, b = (scale_pair(X[tr], X[te]) if scaling == 'train' else (X[tr], X[te]))
        pred = RidgeClassifier(alpha=1.0).fit(a, y[tr]).predict(b)
        score = (float(balanced_accuracy_score(y[te], pred))
                 if len(np.unique(y[te])) == len(np.unique(y)) else None)
        folds.append({'fold':fold,'n_train':len(tr),'n_test':len(te),'balanced_accuracy':score})
        ys.extend(y[te].tolist()); ps.extend(pred.tolist()); indices.extend(te.tolist())
    return {'balanced_accuracy':float(balanced_accuracy_score(ys, ps)),
            'mean_fold_balanced_accuracy':float(np.mean([f['balanced_accuracy'] for f in folds]))
                if all(f['balanced_accuracy'] is not None for f in folds) else None,
            'folds':folds, 'test_indices':indices, 'truth':ys, 'predictions':ps}


def grouped_splits(groups):
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]


def category_splits(info, remove_train_self=False, remove_test_self=False):
    x, y = info[:,0], info[:,1]
    same_input = x == y
    result=[]
    for held in range(4):
        test=(x//2==held)|(y//2==held)
        train=~test
        if remove_train_self: train &= ~same_input
        if remove_test_self: test &= ~same_input
        tr,te=np.flatnonzero(train),np.flatnonzero(test)
        if set(map(tuple,info[tr])) & set(map(tuple,info[te])):
            raise ValueError('Input pairs overlap across the category split')
        result.append((tr,te))
    return result


def category_metrics(X, y, info, remove_train_self=False, remove_test_self=False):
    splits=category_splits(info,remove_train_self,remove_test_self)
    d=evaluate(X,y,splits,'none')
    for fold,(tr,te) in zip(d['folds'],splits):
        p=RidgeClassifier(alpha=1.0).fit(X[tr],y[tr]).predict(X[te])
        fold['confusion_matrix_labels_0_1']=confusion_matrix(y[te],p,labels=[0,1]).tolist()
        fold['n_unique_test_pairs']=len(set(map(tuple,info[te])))
    return d


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False) + '\n', encoding='utf-8')


def load_frames(deposit, experiment, bin_size, n_tokens, cache):
    """Read original float32 camera frames. No fitted transformation here."""
    # Bind caches to the data root as well as the experiment and bin size.
    source_key = hashlib.sha256(str(deposit.resolve()).encode()).hexdigest()[:12]
    cachefile=cache/f'{experiment}_bin{bin_size}_{source_key}.npz'
    if cachefile.exists():
        z=np.load(cachefile, allow_pickle=False)
        return ({i:z[f's{i}'] for i in range(n_tokens)},
                {(y,p):z[f'b{y}_{p}'] for y in range(n_tokens) for p in range(4)},
                {(x,y,p):z[f'p{x}_{y}_{p}'] for x in range(n_tokens) for y in range(n_tokens) for p in range(4)})
    root=deposit/'shared_raw_data/result4_frame_data'/experiment
    if experiment=='exp_bilinear_001': root=root/'exp_bilinear_001'
    labels=['0','pi2','pi','3pi2']
    def read(sub,pattern):
        arrays=[]
        for r in range(5):
            frame = root/sub/pattern.format(r=r)
            with frame.open('rb') as stream:
                if stream.read(43).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise ValueError(f'{frame} is a Git LFS pointer. Run git lfs pull first.')
            a=np.load(frame, allow_pickle=False)
            h,w=a.shape; nh,nw=h//bin_size,w//bin_size
            arrays.append(a[:nh*bin_size,:nw*bin_size].reshape(nh,bin_size,nw,bin_size).mean(axis=(1,3)).reshape(-1))
        return np.array(arrays)
    s={t:read('singles',f'single_t{t}_rep{{r:02d}}.npy') for t in range(n_tokens)}
    b={(y,p):read('blank_controls',f'blank_y{y}_phi{labels[p]}_rep{{r:02d}}.npy') for y in range(n_tokens) for p in range(4)}
    q={(x,y,p):read('pair_interactions',f'pair_x{x}_y{y}_phi{labels[p]}_rep{{r:02d}}.npy') for x in range(n_tokens) for y in range(n_tokens) for p in range(4)}
    payload={f's{k}':v for k,v in s.items()}
    payload.update({f'b{y}_{p}':v for (y,p),v in b.items()})
    payload.update({f'p{x}_{y}_{p}':v for (x,y,p),v in q.items()})
    np.savez_compressed(cachefile,**payload)
    return s,b,q


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deposit', type=Path, default=Path(__file__).resolve().parents[1] / 'source-materials')
    parser.add_argument('--output',type=Path,default=Path('outputs/result4'))
    parser.add_argument('--stage',choices=['main','refined','all'],default='main')
    args=parser.parse_args()
    dep=args.deposit.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if out == dep or dep in out.parents:
        parser.error('Choose an output directory outside the archived source materials.')
    cache=out/'frame_cache'; cache.mkdir(exist_ok=True)
    threadpool_limits(limits=1)
    start=time.time()
    sem=module(dep/'04_result4_paper/scripts/analyze_complex_B_semantic.py','original_semantic')
    xor=module(dep/'04_result4_paper/scripts/analyze_complex_B_xor_showcase.py','original_xor')
    adv=module(dep/'05_result4_refine_paper/scripts/analyze_advantage_boundary.py','original_advantage')
    sparse=module(dep/'05_result4_refine_paper/scripts/analyze_three_bin_sparse.py','original_sparse')
    environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                 'scikit_learn':sklearn.__version__,'ridge_alpha':1.0,'started_utc':datetime.now(timezone.utc).isoformat(),
                 'source':'https://zenodo.org/records/19890402',
                 'archive_md5':'5b742fb14f82496aa6fda35ad7613c80'}
    environment['stage'] = args.stage
    environment['input_sha256'] = {
        str(p.relative_to(dep)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [dep/'provenance/checksums_sha256.txt',
                  dep/'04_result4_paper/scripts/analyze_complex_B_semantic.py',
                  dep/'04_result4_paper/scripts/analyze_complex_B_xor_showcase.py',
                  dep/'05_result4_refine_paper/scripts/analyze_advantage_boundary.py',
                  dep/'05_result4_refine_paper/scripts/analyze_three_bin_sparse.py']
    }
    environment['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(out/'environment.json',environment)
    summary=[]
    if args.stage in ['main','all']:
        for name,mod,nt in [('semantic',sem,8),('xor',xor,4)]:
            experiment='exp_complex_B_semantic' if nt==8 else 'exp_bilinear_001'
            s,b,p=load_frames(dep,experiment,25,nt,cache)
            B=mod.compute_per_repeat_B(b,p)
            features,*ys=mod.build_feature_families({t:s[t].mean(axis=0) for t in range(nt)},B)
            pubfile=('complex_B_semantic_results/complex_B_semantic_results.json' if nt==8 else 'complex_B_xor_showcase_results/complex_B_xor_results.json')
            published=json.loads((dep/'04_result4_paper/source_data/processed'/pubfile).read_text())
            tasks=['A','B','C'] if nt==8 else ['A','B']
            keys=['task_A_pair_id','task_B_same_cat','task_C_cat_pair'] if nt==8 else ['full_task_A_bal_acc','full_task_B_bal_acc']
            rows={}; samples=len(ys[0]); loo=list(LeaveOneOut().split(np.arange(samples)))
            np.savez_compressed(out/f'{name}_bin25_features.npz',**features,**{f'label_{t}':y for t,y in zip(tasks,ys)})
            for family,X in features.items():
                rows[family]={}
                for task,y,key in zip(tasks,ys,keys):
                    # Original function, original global preprocessing.
                    Xglobal,_=scale_pair(X,X)
                    original=mod.evaluate_loocv(Xglobal,y)
                    score=original[0] if nt==8 else original[1]
                    corrected=evaluate(X,y,loo,'train')
                    printed=published['classification_results'][family][key]
                    rows[family][task]={'published':printed,'reproduced':score,'corrected':corrected,
                                       'replication_error':abs(score-printed)}
                    summary.append({'experiment':name,'protocol':'repeat_LOO','family':family,'task':task,
                                    'published':printed,'reproduced':score,'corrected':corrected['balanced_accuracy']})
                    print(name,family,task,'original',round(score,6),'train-only',round(corrected['balanced_accuracy'],6),flush=True)
                dump(out/f'{name}_main.json',rows)
            info=np.array([(x,y) for x in range(nt) for y in range(nt) for _ in range(5)])
            pid=ys[0]; reps=np.tile(np.arange(5),nt*nt)
            unordered=np.minimum(info[:,0],info[:,1])*nt+np.maximum(info[:,0],info[:,1])
            supplemental={}
            for family in ['zx_zy_concat','digital_bilinear_ew','complex_B']:
                X=features[family]; supplemental[family]={}
                for task,y in zip(tasks,ys):
                    d={'repeat_block':evaluate(X,y,grouped_splits(reps),'train')}
                    if task!='A':
                        d['ordered_pair_holdout']=evaluate(X,y,grouped_splits(pid),'train')
                        d['unordered_pair_holdout']=evaluate(X,y,grouped_splits(unordered),'train')
                    supplemental[family][task]=d
                print(name,family,'additional split comparisons complete',flush=True)
            if nt==8:
                rng=np.random.default_rng(42); held=[]
                for cx in range(4):
                    for cy in range(4):held.append((int(rng.choice([2*cx,2*cx+1])),int(rng.choice([2*cy,2*cy+1]))))
                te=np.array([tuple(p) in held for p in info]); split=[(np.flatnonzero(~te),np.flatnonzero(te))]
                assert not set(map(tuple,info[~te])) & set(map(tuple,info[te]))
                for family,X in features.items():
                    old=evaluate(X,ys[1],split,'global'); fixed=evaluate(X,ys[1],split,'train')
                    pub=published['cross_split_results'][family]['cross_split_same_cat_bal_acc']
                    supplemental.setdefault(family,{})['D_fixed16']={'published':pub,'original':old,'corrected':fixed,'held_pairs':held}
                    summary.append({'experiment':name,'protocol':'fixed_16_pair_holdout','family':family,'task':'D','published':pub,'reproduced':old['balanced_accuracy'],'corrected':fixed['balanced_accuracy']})
                    print('main D',family,old['balanced_accuracy'],fixed['balanced_accuracy'],flush=True)
            dump(out/f'{name}_split_comparisons.json',supplemental)
        with (out/'main_comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    if args.stage in ['refined','all']:
        published=json.loads((dep/'05_result4_refine_paper/source_data/processed/three_bin_sparse/three_bin_sparse_results.json').read_text())
        # Capture the exact released raw-input construction, including its RNG draw order.
        oldbench=sparse.bench
        sparse.bench=lambda X,pid,sc,cp,info,name:(name,X,pid,sc,cp,np.array(info))
        raw=sparse.raw_input_direct(); sparse.bench=oldbench
        rawrows={}
        for name,X,pid,sc,cp,info in raw:
            pub=next(r for r in published['raw_input_direct'] if r['name']==name)
            base=sparse.bench(X,pid,sc,cp,info,name)
            rawrows[name]={'published':pub,'reproduced':base,'D_no_self':category_metrics(X,sc,info,True,True),
                           'D_nonself_test_only':category_metrics(X,sc,info,False,True)}
        dump(out/'raw_baselines.json',rawrows)
        sparse_rows=[]; fixed_rows={}
        for bin_size in [25,10,5]:
            s,b,p=load_frames(dep,'exp_complex_B_semantic',bin_size,8,cache)
            B=sem.compute_per_repeat_B(b,p)
            features,pid,sc,cp,pinfo=adv.build_features(s,B);info=np.array(pinfo)
            if bin_size in [25,10]:
                fixed_rows[str(bin_size)]={}
                for family,X in features.items():
                    fixed_rows[str(bin_size)][family]={'reproduced':adv.run_benchmark(X,pid,sc,cp,pinfo,family),
                         'D_all':category_metrics(X,sc,info), 'D_no_self':category_metrics(X,sc,info,True,True),
                         'D_nonself_test_only':category_metrics(X,sc,info,False,True),
                         'B_grouped':evaluate(X,sc,grouped_splits(pid),'none'),
                         'C_grouped':evaluate(X,cp,grouped_splits(pid),'none')}
                dump(out/'refined_full.json',fixed_rows)
            nh=500//bin_size
            for nout in [4,9,16,25,36,49,64,100,144]:
                idx,coords=sparse.sparse_tap_indices(nh,nh,nout,max(1,int(np.ceil(10.5/bin_size))))
                # Same sparse channels and feature ordering as the original script.
                feats,pp,ss,cc,ii=sparse.build_optical_at_taps(s,B,idx)
                pub=next(a for a in published['anchors'] if a['bin_size']==bin_size)
                pub=next(r for r in pub['sweep'] if r['n_out']==nout)
                for family,X in feats.items():
                    original=sparse.bench(X,pp,ss,cc,ii,family)
                    all_result=category_metrics(X,ss,np.array(ii))
                    ablation=category_metrics(X,ss,np.array(ii),True,True)
                    conditional=category_metrics(X,ss,np.array(ii),False,True)
                    sparse_rows.append({'bin':bin_size,'n_out':nout,'family':family,
                        'published':pub[family], 'reproduced':original,
                        'D_all':all_result,'D_no_self':ablation,'D_nonself_test_only':conditional,
                        'tap_indices':idx.tolist(), 'tap_coordinates':coords})
                print('sparse',bin_size,nout,'B D',original['D'],'no self',round(ablation['mean_fold_balanced_accuracy'],6),flush=True)
                dump(out/'sparse_comparisons.json',sparse_rows)
        identity=[]
        for tr,te in category_splits(info):
            pred=(info[te,0]==info[te,1]).astype(int)
            identity.append(float(balanced_accuracy_score(sc[te],pred)))
        dump(out/'identity_control.json',{'category_fold_balanced_accuracy':identity,'mean':float(np.mean(identity)),
             'interpretation':'Task-structure control using known input identities, not an operator-matched optical readout baseline.'})
        with (out/'sparse_summary.csv').open('w') as f:
            fields=['bin','n_out','family','published_D','reproduced_D','no_self_D','nonself_test_only_D']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in sparse_rows:w.writerow({'bin':r['bin'],'n_out':r['n_out'],'family':r['family'],
                'published_D':r['published']['D'],'reproduced_D':r['reproduced']['D'],
                'no_self_D':r['D_no_self']['mean_fold_balanced_accuracy'],
                'nonself_test_only_D':r['D_nonself_test_only']['mean_fold_balanced_accuracy']})
    environment['elapsed_seconds'] = round(time.time()-start, 3)
    dump(out/'environment.json', environment)
    print('Completed in',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':main()
