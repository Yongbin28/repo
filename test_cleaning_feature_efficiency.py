"""Controlled cleaning/representation ablations, independent of production models.

Run from the repository: python test_cleaning_feature_efficiency.py
"""
from pathlib import Path
from time import perf_counter
import json
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from waferpulse.tools import equipment_data as ed
from waferpulse.experiments.bosch_plasma_etch_benchmark import group_to_experiment_key, _safe_name, _slope

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "cleaning_feature_efficiency"
SEEDS = (42, 123)
TOLERANCE = 0.02  # Two absolute percentage points, chosen before evaluating.


def equipment():
    start = perf_counter()
    response, _ = ed._deduplicate_response(ed._read_semicolon_csv(ROOT / "EquipmentData/response.csv"), .75)
    frames, naive_frames, flags, quality = {}, {}, {}, {}
    identity = response[["lot", "wafer"]].copy()
    for stage in ("equipment1", "equipment2"):
        raw = ed._read_semicolon_csv(ROOT / f"EquipmentData/{stage}.csv")
        numeric, _, summary = ed._inspect_equipment(raw, stage, 1.0)
        frames[stage] = numeric
        naive = numeric[["lot", "wafer", "_timestamp_number"]].copy()
        formatted = naive.copy()
        for sensor in [c for c in numeric if c.startswith("sensor_")]:
            naive[sensor] = pd.to_numeric(raw[sensor], errors="coerce")
            formatted[sensor] = (raw[sensor].notna() & naive[sensor].isna()).astype(float)
        naive_frames[stage], flags[stage], quality[stage] = naive, formatted, summary
        identity = identity.merge(numeric[["lot", "wafer"]].drop_duplicates(), on=["lot", "wafer"], validate="one_to_one")
    identity = identity.reset_index(drop=True)
    targets = identity.merge(response, on=["lot", "wafer"], validate="one_to_one")
    input_seconds = perf_counter() - start
    feature_sets, feature_seconds = {}, {}
    for name, source in (("minimal_summary", naive_frames), ("recovered_summary", frames)):
        start = perf_counter()
        tables = [ed._aggregate_temporal_features(source[stage], stage) for stage in source]
        merged = tables[0].merge(tables[1], on=["lot", "wafer"], validate="one_to_one")
        merged = identity.merge(merged, on=["lot", "wafer"], validate="one_to_one")
        feature_sets[name] = merged.drop(columns=["lot", "wafer"])
        feature_seconds[name] = perf_counter() - start
    start = perf_counter()
    raw_parts, flag_parts, names = [], [], []
    for stage in frames:
        tensor, sensor_names = ed._trace_tensor(frames[stage], identity, stage)
        flag_tensor, _ = ed._trace_tensor(flags[stage], identity, stage)
        raw_parts.append(tensor)
        flag_parts.append(flag_tensor)
        names.extend(sensor_names)
    raw = np.concatenate(raw_parts, axis=1)
    flag_array = np.concatenate(flag_parts, axis=1)
    feature_sets["cleaned_trace"] = pd.DataFrame(raw.reshape(len(identity), -1), columns=[f"{name}__t{t}" for name in names for t in range(176)])
    feature_seconds["cleaned_trace"] = perf_counter() - start
    feature_sets["cleaned_summary"] = feature_sets["recovered_summary"]
    feature_seconds["cleaned_summary"] = feature_seconds["recovered_summary"]
    rates = pd.DataFrame(flag_array.mean(axis=2), columns=names)
    return dict(name="EquipmentData", X=feature_sets, transform_seconds=feature_seconds, input_seconds=input_seconds, identity=identity, y=targets.response.to_numpy(), labels=targets['class'].eq('bad').astype(int).to_numpy(), groups=identity.lot.to_numpy(), rates=rates, quality=quality)


def bosch():
    source = ROOT / "dataset/bosch_plasma_etch"
    start = perf_counter()
    with Dataset(source / "Dictionary_process.nc") as file:
        decoder = np.asarray(file.variables["data"][:], dtype=float)
    traces = {}
    with Dataset(source / "Process_data.nc") as file:
        for name, group in file.groups.items():
            traces[group_to_experiment_key(name)] = (decoder[np.asarray(group.variables['data'][:], dtype=np.uint16)], np.asarray(group.variables['feature'][:], dtype=str), np.asarray(group.variables['times'][:], dtype=float))
    measurement = pd.read_csv(source / "Si_Oxide_etch_89_points.csv")
    target = measurement.groupby(["experiment_key", "lot_number"], as_index=False).si_etch.mean()
    target = target.loc[target.experiment_key.isin(traces)].sort_values('experiment_key').reset_index(drop=True)
    input_seconds = perf_counter() - start
    start = perf_counter()
    rows = []
    for key in target.experiment_key:
        values, names, times = traces[key]
        row = {"process__sample_count":len(times), "process__duration":float(times[-1]-times[0])}
        for i, name in enumerate(names):
            signal = values[:, i]
            signal = signal[np.isfinite(signal)]
            prefix = "signal__" + _safe_name(name)
            stats = dict(mean=np.mean(signal), std=np.std(signal, ddof=1), min=np.min(signal), max=np.max(signal), q25=np.quantile(signal,.25), median=np.median(signal), q75=np.quantile(signal,.75), first=signal[0], last=signal[-1], slope=_slope(signal))
            row.update({prefix+'__'+stat:float(value) for stat,value in stats.items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_seconds = perf_counter() - start
    start = perf_counter()
    all_names = sorted(set(name for _, names, _ in traces.values() for name in names))
    rows=[]
    for key in target.experiment_key:
        values, names, times=traces[key]
        grid=np.linspace(times[0],times[-1],256)
        channels={name:np.interp(grid,times,values[:,i]) for i,name in enumerate(names)}
        rows.append(np.concatenate([channels.get(name,np.full(256,np.nan)) for name in all_names]))
    raw=pd.DataFrame(rows,columns=[f"signal__{_safe_name(name)}__t{t}" for name in all_names for t in range(256)])
    raw_seconds=perf_counter()-start
    return dict(name="Bosch",X={"minimal_summary":summary,"cleaned_summary":summary,"cleaned_trace":raw},transform_seconds={"minimal_summary":summary_seconds,"cleaned_summary":summary_seconds,"cleaned_trace":raw_seconds},input_seconds=input_seconds,identity=target[['experiment_key','lot_number']],y=target.si_etch.to_numpy(),labels=target.si_etch.gt(44).astype(int).to_numpy(),groups=target.lot_number.to_numpy(),rates=None)


def evaluate(dataset):
    name=dataset['name']
    groups,y,labels=dataset['groups'],dataset['y'],dataset['labels']
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=42) if name=='EquipmentData' else GroupKFold(5)
    splits=list(splitter.split(dataset['identity'],labels,groups))
    rows,folds,ledgers=[],[],[]
    for variant,X in dataset['X'].items():
        for seed in SEEDS:
            prediction=np.full(len(y),np.nan)
            train_seconds=0.0
            predict_seconds=0.0
            sizes=[]
            for fold,(train,test) in enumerate(splits,1):
                assert set(groups[train]).isdisjoint(groups[test])
                timer=perf_counter()
                columns=list(X.columns)
                if variant.startswith('cleaned') and name=='EquipmentData':
                    rate=dataset['rates'].iloc[train].mean()
                    sensors=set(rate.index[rate.le(.01)])
                    columns=[c for c in columns if c.rsplit('__',1)[0] in sensors]
                elif variant.startswith('cleaned') and name=='Bosch':
                    # Fold-local equivalent of >=20 observations / 96 source wafers.
                    observed=X.iloc[train].notna().mean()
                    columns=observed.index[observed.ge(20/96)].tolist()
                if name=='EquipmentData':
                    estimator=RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=1)
                    train_y=labels[train]
                else:
                    estimator=LGBMRegressor(n_estimators=300,learning_rate=.04,num_leaves=15,min_child_samples=15,reg_lambda=2.,verbosity=-1,n_jobs=1,random_state=seed)
                    train_y=y[train]
                model=make_pipeline(SimpleImputer(strategy='median'),VarianceThreshold(),estimator)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='Skipping features without any observed values')
                    model.fit(X.iloc[train][columns],train_y)
                elapsed=perf_counter()-timer
                train_seconds+=elapsed
                timer=perf_counter()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='Skipping features without any observed values')
                    if name=='EquipmentData':
                        prediction[test]=model.predict_proba(X.iloc[test][columns])[:,1]
                        predicted=prediction[test]>=.5
                    else:
                        prediction[test]=model.predict(X.iloc[test][columns])
                        predicted=prediction[test]>44
                predict_seconds+=perf_counter()-timer
                feature_count=int(model.named_steps['variancethreshold'].get_support().sum())
                sizes.append(feature_count)
                folds.append(dict(dataset=name,variant=variant,seed=seed,fold=fold,training_seconds=elapsed,features=feature_count,accuracy=accuracy_score(labels[test],predicted),f1=f1_score(labels[test],predicted,zero_division=0),recall=recall_score(labels[test],predicted,zero_division=0)))
            predicted=prediction>=.5 if name=='EquipmentData' else prediction>44
            tn,fp,fn,tp=confusion_matrix(labels,predicted,labels=[0,1]).ravel()
            row=dict(dataset=name,variant=variant,seed=seed,rows=len(y),input_features=X.shape[1],mean_fitted_features=float(np.mean(sizes)),accuracy=accuracy_score(labels,predicted),balanced_accuracy=balanced_accuracy_score(labels,predicted),f1=f1_score(labels,predicted,zero_division=0),recall=recall_score(labels,predicted,zero_division=0),precision=precision_score(labels,predicted,zero_division=0),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),input_seconds=dataset['input_seconds'],representation_seconds=dataset['transform_seconds'][variant],cv_training_seconds=train_seconds,prediction_seconds=predict_seconds,total_preparation_training_seconds=dataset['input_seconds']+dataset['transform_seconds'][variant]+train_seconds)
            if name=='Bosch':
                row.update(r2=r2_score(y,prediction),rmse=float(np.sqrt(mean_squared_error(y,prediction))))
            rows.append(row)
            ledgers.append(dataset['identity'].assign(dataset=name,variant=variant,seed=seed,actual_response=y,actual_class=labels,predicted_value=prediction,predicted_class=predicted.astype(int)))
            print(f"{name} {variant} seed={seed}: accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}, training={train_seconds:.2f}s",flush=True)
            pd.DataFrame(rows).to_csv(OUT/f'{name}_metrics.csv',index=False)
            pd.DataFrame(folds).to_csv(OUT/f'{name}_folds.csv',index=False)
            pd.concat(ledgers,ignore_index=True).to_csv(OUT/f'{name}_predictions.csv',index=False)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(validation="Five identical lot-held-out folds within each dataset; two fixed model seeds; no tuning",accuracy_tolerance=TOLERANCE,f1_tolerance=TOLERANCE,tolerance_note="Exploratory tolerance chosen before results; meeting it does not prove statistical equivalence",equipment_model="100-tree balanced RandomForestClassifier, max_depth12, leaf2, sqrt features; threshold0.5",bosch_model="300-estimator LightGBMRegressor; predictions >44 form research high-etch class",bosch_trace_baseline="All available process channels interpolated to256 normalized-time positions per wafer; no spatial coordinates",cleaning="EquipmentData format recovery and <=1% source-format inclusion learned on training wafers; Bosch >=20/96 observed-column eligibility learned on training wafers",common_preprocessing="Median imputation and variance removal fitted on training folds in every branch",timing="Measured locally in one sequential run; two seeds are not independent test cohorts. Input decoding shared across branches; representation preparation measured once; fit includes fold-local eligibility/imputation/variance/model. Timings include all5 folds.",claim_limits="Mandatory identity/label validation retained in all branches. No industrial equivalence claim. Feature comparisons may alter effective tree feature sampling. Models are fixed experimental controls, not an exact reproduction of saved champions.",source_sha256={str(p.relative_to(ROOT)):ed._sha256(p) for p in [ROOT/'EquipmentData/equipment1.csv',ROOT/'EquipmentData/equipment2.csv',ROOT/'EquipmentData/response.csv',ROOT/'dataset/bosch_plasma_etch/Process_data.nc',ROOT/'dataset/bosch_plasma_etch/Dictionary_process.nc',ROOT/'dataset/bosch_plasma_etch/Si_Oxide_etch_89_points.csv']})
    (OUT/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    rows=[]
    for prepare in (equipment,bosch):
        print(f"Preparing {prepare.__name__}",flush=True)
        dataset=prepare()
        rows.extend(evaluate(dataset))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)


if __name__=='__main__':
    main()
    from summarize_feature_efficiency import main as summarize
    summarize()
