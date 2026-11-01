"""Summarize paired experiments and bootstrap frozen predictions by lot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/cleaning_feature_efficiency'


def main():
    metrics=pd.read_csv(OUT/'metrics.csv')
    summary=metrics.groupby(['dataset','variant'],as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT/'summary.csv',index=False)
    contrasts=[]
    for dataset in ('EquipmentData','Bosch'):
        ledger=pd.read_csv(OUT/f'{dataset}_predictions.csv')
        group_column='lot' if dataset=='EquipmentData' else 'lot_number'
        groups=sorted(ledger[group_column].unique())
        seeds=sorted(ledger.seed.unique())
        rng=np.random.default_rng(20261008)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
        def counts(variant):
            arrays=[]
            for seed in seeds:
                part=ledger.loc[ledger.variant.eq(variant)&ledger.seed.eq(seed)]
                arrays.append([[int(((g.actual_class==a)&(g.predicted_class==p)).sum()) for a,p in [(0,0),(0,1),(1,0),(1,1)]] for name in groups for g in [part.loc[part[group_column].eq(name)]]])
            return np.asarray(arrays)
        def bootstrap_scores(count):
            total=np.einsum('bg,sgc->bsc',weights,count)
            tn,fp,fn,tp=[total[:,:,i] for i in range(4)]
            accuracy=(tn+tp)/total.sum(axis=2)
            denom=2*tp+fp+fn
            f1=np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
            return accuracy.mean(axis=1),f1.mean(axis=1)
        comparisons=[('cleaning','minimal_summary','cleaned_summary'),('feature_extraction','cleaned_trace','cleaned_summary')]
        if dataset=='EquipmentData':
            comparisons.extend([('format_recovery','minimal_summary','recovered_summary'),('sensor_exclusion','recovered_summary','cleaned_summary')])
        for label,before,after in comparisons:
            a=summary.loc[summary.dataset.eq(dataset)&summary.variant.eq(before)].iloc[0]
            b=summary.loc[summary.dataset.eq(dataset)&summary.variant.eq(after)].iloc[0]
            a_boot=bootstrap_scores(counts(before))
            b_boot=bootstrap_scores(counts(after))
            row=dict(dataset=dataset,comparison=label,before=before,after=after,accuracy_before=a.accuracy,accuracy_after=b.accuracy,accuracy_difference_pp=100*(b.accuracy-a.accuracy),f1_before=a.f1,f1_after=b.f1,training_before_seconds=a.cv_training_seconds,training_after_seconds=b.cv_training_seconds,training_reduction_percent=100*(1-b.cv_training_seconds/a.cv_training_seconds),total_before_seconds=a.total_preparation_training_seconds,total_after_seconds=b.total_preparation_training_seconds,total_reduction_percent=100*(1-b.total_preparation_training_seconds/a.total_preparation_training_seconds))
            for index,metric in enumerate(('accuracy','f1')):
                lower,upper=np.quantile(b_boot[index]-a_boot[index],[.025,.975])
                row[metric+'_difference_ci_low']=float(lower)
                row[metric+'_difference_ci_high']=float(upper)
            row['observed_accuracy_f1_within_tolerance']=bool(b.accuracy>=a.accuracy-.02 and b.f1>=a.f1-.02)
            row['bootstrap_accuracy_f1_lower_bounds_within_tolerance']=bool(row['accuracy_difference_ci_low']>=-.02 and row['f1_difference_ci_low']>=-.02)
            contrasts.append(row)
    contrast=pd.DataFrame(contrasts)
    contrast.to_csv(OUT/'comparisons.csv',index=False)
    lines=['# Cleaning and feature extraction experiment — 8 October 2026','','Two fixed model seeds, five identical held-out-lot folds per dataset. Results below average the two seeds. These are controlled experiments, not replacements for production champion metrics.','','## Findings','']
    for _,r in contrast.iterrows():
        if r.comparison in ('cleaning','feature_extraction'):
            lines.append(f"- {r.dataset}, {r.comparison}: accuracy {r.accuracy_before:.2%} → {r.accuracy_after:.2%}; bad/high-etch F1 {r.f1_before:.4f} → {r.f1_after:.4f}; five-fold training {r.training_before_seconds:.3f}s → {r.training_after_seconds:.3f}s; total preparation and training {r.total_before_seconds:.3f}s → {r.total_after_seconds:.3f}s.")
    lines.extend(['','The tested summary representations train faster with slightly higher observed classification accuracy. Cleaning alone has negligible classification-accuracy benefit here. Paired bootstrap intervals do not establish accuracy/F1 non-inferiority within the recorded two-point margin for feature extraction on either dataset. A defensible Chapter 5 statement is: "On the evaluated held-out lots, summary-feature models reduced training time while achieving comparable or slightly higher observed classification scores than trace-input controls." Do not generalize this to guaranteed maintenance of production accuracy.','', 'For EquipmentData, always predicting good gives 82.49% accuracy but zero bad-class F1. The balanced experimental classifier has lower overall accuracy and detects some bad wafers. Accuracy alone is therefore insufficient evidence of training quality. Bosch always predicting within-threshold gives 76.14% accuracy.','','## Results','','| Dataset | Input | Accuracy | Bad/high-etch F1 | Recall | Five-fold training (s) | Total preparation + training (s) |','|---|---|---:|---:|---:|---:|---:|'])
    for _,r in summary.iterrows():
        lines.append(f"| {r.dataset} | {r.variant} | {r.accuracy:.2%} | {r.f1:.4f} | {r.recall:.2%} | {r.cv_training_seconds:.3f} | {r.total_preparation_training_seconds:.3f} |")
    lines.extend(['','## Paired comparisons','','| Dataset | Comparison | Accuracy change (percentage points) | F1 change | Training-time reduction | Total-time reduction |','|---|---|---:|---:|---:|---:|'])
    for _,r in contrast.iterrows():
        lines.append(f"| {r.dataset} | {r.comparison} | {r.accuracy_difference_pp:+.2f} | {r.f1_after-r.f1_before:+.4f} | {r.training_reduction_percent:+.1f}% | {r.total_reduction_percent:+.1f}% |")
    lines.extend(['','## Interpretation and limitations','','- Cleaning comparisons preserve the summary representation. Feature comparisons preserve cleaning eligibility rules. All branches fit median imputation and variance removal using training lots only. Mandatory key/label validation remains in the minimally prepared branch.','- EquipmentData uses a 100-tree balanced Random Forest at probability threshold 0.5. Bosch uses a fixed 300-tree LightGBM regressor followed by the research etch threshold >44.0. Parameters are fixed across variants. No score-based model or threshold selection is performed.','- EquipmentData retains all 176 time readings per available sensor in the trace comparison. Bosch traces use 256 interpolated normalized-time readings per channel because traces have unequal lengths. It is not a comparison with every original Bosch reading.','- Bosch summaries use ten sensor statistics plus sample count/duration, compatible with the original summary cache representation. This is not the latest expanded statistical extractor or an exact reproduction of the saved 88.64% screening champion.','- Timing includes five folds, not one final fit. Input decoding is shared across representations and included equally in total times; representation preparation is measured separately. Eligibility checks, imputation, variance removal and estimator fitting are included in training time. Timing is one local sequential run with two seeds, not a hardware-independent benchmark. Shared input parsing performs recovery even for the minimal branch, so total times do not isolate recovery overhead.','- A 2-percentage-point accuracy/F1 tolerance was recorded before seeing results. Observed scores within tolerance are only exploratory evidence. The 2,000 paired lot-bootstrap intervals in comparisons.csv resample frozen out-of-fold predictions; they do not retrain models or prove statistical equivalence.','- Tree feature sampling and split selection can change when the representation changes, even with identical model parameters. The timing/score conclusion applies to the tested pipeline and population.','- Two seeds reuse the same wafers and folds; they are not independent validation cohorts. Small gains should not be described as established improvements.','- Feature extraction cannot be called universally accuracy-preserving: inspect accuracy, F1, recall and paired uncertainty together. Nor does faster estimator fitting imply faster total processing.','- Source hashes, configuration, per-fold metrics and per-wafer predictions are saved in the same output directory. Production models and the Chapter 5 draft were not changed.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary[['dataset','variant','accuracy','f1','cv_training_seconds','total_preparation_training_seconds']].to_string(index=False))
    print(contrast[['dataset','comparison','accuracy_difference_pp','training_reduction_percent','total_reduction_percent','accuracy_difference_ci_low','f1_difference_ci_low']].to_string(index=False))


if __name__=='__main__':
    main()
