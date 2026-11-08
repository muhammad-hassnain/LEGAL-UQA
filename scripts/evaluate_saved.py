"""Evaluate saved dataset and model outputs without model/API calls.

Run from any directory: python -m scripts.evaluate_saved
Outputs JSON provenance and results under output/evaluation by default.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import string
import unicodedata

import nltk
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
import pandas as pd
import pyarrow as pa
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'claude-3.5-sonnet': ('Claude-3.5-Sonnet', 'claude-3.5-sonnet_human_eval.xlsx'),
    'gemini-1.5-flash-002': ('Gemini-1.5-Flash-002', 'gemini-flash_human_eval.xlsx'),
    'gemini-1.5-pro-002': ('Gemini-1.5-Pro-002', 'gemini-pro_human_eval.xlsx'),
    'mistral-large': ('Mistral-Large', 'mistral-large_human_eval.xlsx'),
    'mt5-large': ('mT5-large-UQA-1.0', 'mt5-large_human_eval.xlsx'),
    'gpt-5.4': ('GPT-5.4', 'gpt-5.4-responses.csv'),
    'deepseek-v4-pro': ('DeepSeek-V4-Pro', 'deepseek-v4-pro-responses.csv'),
    'claude-sonnet-5': ('Claude-Sonnet-5', 'claude-sonnet-5-responses.csv'),
    'gemma-4-e4b': ('Gemma-4-E4B', 'gemma-4-e4b-responses.csv'),
}
UPDATED = list(MODELS)[-4:]


def load_split(split: str) -> pd.DataFrame:
    path = ROOT / 'data/complete_dataset' / split / 'data-00000-of-00001.arrow'
    with pa.memory_map(str(path), 'r') as source:
        return pa.ipc.open_stream(source).read_all().to_pandas()


def squad_normalize(text: str) -> str:
    """SQuAD v1 normalization, preserved for comparison with historical scores."""
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def urdu_normalize(text: str) -> str:
    """Explicit sensitivity analysis; does not change stored references."""
    text = unicodedata.normalize('NFC', text).translate(str.maketrans({'ك': 'ک', 'ي': 'ی', 'ى': 'ی'}))
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith('M') or category == 'Cf' or char == '\u0640':
            continue
        if category.startswith('P'):
            chars.append(' ')
        elif category == 'Nd':
            chars.append(str(unicodedata.decimal(char)))
        else:
            chars.append(char.lower())
    return ' '.join(''.join(chars).split())


def token_f1(prediction: str, reference: str, normalize=squad_normalize) -> float:
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2 * overlap / (len(p) + len(r))


def annotation_text(text: str) -> str:
    """Ignore layout-only list markers/whitespace in archived Excel copies."""
    return ' '.join(re.sub(r'(?<!\S)[-*](?!\S)', '', text).split())


def annotation_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path) if path.suffix == '.xlsx' else pd.read_csv(path)
    df = df.dropna(how='all').copy()
    score = next(c for c in df.columns if c.lower() == 'score')
    df = df.rename(columns={score: 'Score'})
    df['id'] = df['id'].astype(int)
    if not df['id'].is_unique or df['Score'].isna().any() or not df['Score'].isin([0,.5,1]).all():
        raise ValueError(f'Invalid annotation IDs/scores: {path}')
    return df.set_index('id').sort_index()


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_human_scores() -> dict:
    """Refresh the four updated-model human summaries with text/ID checks."""
    path=ROOT/'data/human_evaluations_qa/accuracy_summary.json'
    summary=json.loads(path.read_text())
    summary['calculated_at_utc']=datetime.now(timezone.utc).isoformat()
    for model in summary['models']:
        source=ROOT/model['source_file']
        raw=pd.read_csv(source)
        df=annotation_frame(source)
        response=pd.read_csv(ROOT/'data/model_qa_responses'/f"{model['model']}-responses.csv").set_index('id').sort_index()
        prediction=next(c for c in df if c.startswith('prediction_text'))
        response_prediction=next(c for c in response if c.startswith('prediction_text'))
        if df.index.tolist()!=response.index.tolist() or df[prediction].tolist()!=response[response_prediction].tolist() or df.answers.tolist()!=response.answers.tolist():
            raise ValueError(f"Annotation text/ID mismatch: {source}")
        score=df.Score
        model.update(source_sha256=source_hash(source),scored_examples=len(df),correct=int(score.eq(1).sum()),
            partially_correct=int(score.eq(.5).sum()),incorrect=int(score.eq(0).sum()),score_sum=float(score.sum()),
            accuracy_percent=float(100*score.mean()),full_credit_accuracy_percent=float(100*score.eq(1).mean()),
            wholly_empty_rows_excluded=int(raw.isna().all(axis=1).sum()),annotation_ids_and_text_match_saved_predictions=True)
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def analyze(out: Path, seed=25, bootstrap_samples=10000) -> dict:
    summarize_human_scores()
    out.mkdir(parents=True, exist_ok=True)
    train, val = load_split('train'), load_split('validation')
    full = pd.concat([train,val],ignore_index=True)
    val = val.reset_index(drop=True)
    if full.isna().any().any() or full.question_eng.duplicated().any():
        raise ValueError('Dataset contains missing fields or duplicate English questions')
    results = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
               'bootstrap':{'seed':seed,'replicates':bootstrap_samples,'unit':'validation context_index',
                            'interval':'paired article-cluster percentile, 2.5/97.5 percentiles; question-weighted mean'},
               'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','nltk','sacrebleu','scipy','scikit-learn','openpyxl']},
               'sources':{},'models':{},'dataset':{}}
    def record(path):
        results['sources'][str(path.relative_to(ROOT))] = source_hash(path)
    record(Path(__file__))
    record(ROOT/'requirements-evaluation-lock.txt')
    record(ROOT/'data/evaluation_environment.json')
    resource_hashes={'punkt_tab_english':{}}
    punkt_path=Path(str(nltk.data.find('tokenizers/punkt_tab/english/')))
    resource_hashes['punkt_tab_english']={p.name:source_hash(p) for p in sorted(punkt_path.iterdir()) if p.is_file()}
    wordnet=nltk.data.find('corpora/wordnet.zip')
    resource_hashes['wordnet_zip_sha256']=source_hash(Path(str(wordnet.zipfile.filename)))
    expected=json.loads((ROOT/'data/evaluation_environment.json').read_text())['nltk_resources']
    if resource_hashes!=expected:raise ValueError('Cached NLTK resources differ from the recorded evaluation environment')
    results['nltk_resource_hashes']=resource_hashes
    from scripts import lexical_retrieval
    record(Path(lexical_retrieval.__file__))
    record(ROOT/'data/human_evaluations_qa/accuracy_summary.json')
    for split in ['train','validation']:
        record(ROOT/'data/complete_dataset'/split/'data-00000-of-00001.arrow')
    # Recover archived printed retrieval summaries; no embedding/API rerun.
    results['retrieval'] = {}
    for model, filename in [('text-embedding-3-large','openai_rets.ipynb'), ('mistral-embed','mistral_rets.ipynb')]:
        path=ROOT/'notebooks/embeddings_testing'/filename
        record(path)
        notebook=json.loads(path.read_text())
        printed='\n'.join(''.join(output.get('text',[])) for cell in notebook['cells'] for output in cell.get('outputs',[]))
        values={int(k):float(value) for k,value in re.findall(r'Top (\d+) Acc: ([\d.]+)%',printed)}
        if set(values)!={1,3,5}:raise ValueError(f'Missing archived retrieval summaries: {path}')
        results['retrieval'][model]={'recall_percent':values, 'source':str(path.relative_to(ROOT)), 'status':'archived printed notebook output; vectors/rankings not retained'}
    results['lexical_retrieval']=lexical_retrieval.evaluate(full,val,urdu_normalize)
    dataset = results['dataset']
    dataset.update(n=len(full),train_n=len(train),validation_n=len(val),
                   qa_article_count=int(full.context_index.nunique()),train_article_count=int(train.context_index.nunique()),
                   validation_article_count=int(val.context_index.nunique()),
                   shared_article_count=len(set(train.context_index)&set(val.context_index)),
                   validation_questions_with_training_article=int(val.context_index.isin(train.context_index).sum()))
    dataset['lengths']={}
    for column in ['question_eng','question_urdu','answer_eng','answer_urdu','context_eng','context_urdu']:
        # Context lengths use unique article IDs, not repeated QA rows.
        values = full.drop_duplicates('context_index')[column] if column.startswith('context') else full[column]
        counts = values.str.split().str.len().to_numpy()
        dataset['lengths'][column]={'n':len(counts),'mean':float(counts.mean()),'median':float(np.median(counts)),
                                    'sd_population':float(counts.std()),'min':int(counts.min()),'max':int(counts.max())}
    dataset['questions_per_article'] = {str(k):int(v) for k,v in full.groupby('context_index').size().value_counts().sort_index().items()}
    categories_path=ROOT/'data/document_chunks/article_wise_chunks_categorized_english.pkl'
    record(categories_path)
    with categories_path.open('rb') as stream:
        categories=pickle.load(stream)
    english_path=ROOT/'data/augmented_datasets/qa_dataset_eng_v1.csv'
    record(english_path)
    english=pd.read_csv(english_path)
    dataset.update(extracted_chunks=len(categories),raw_qa_rows=len(english),unique_english_questions=int(english.Question.nunique()),
                   chunks_without_qa=sorted(set(range(len(categories)))-set(full.context_index)))
    dataset['categories']={}
    for category in ['Small','Medium','Large']:
        ids=[i for i,(_,c) in enumerate(categories) if c==category]
        counts=np.array([len(categories[i][0].split()) for i in ids])
        retained=full.loc[full.context_index.isin(ids)]
        dataset['categories'][category]={'articles':len(ids),'retained_pairs':len(retained),
                'requested_quota':len(ids)*{'Small':1,'Medium':2,'Large':3}[category],
                'mean':float(counts.mean()),'sd_population':float(counts.std()),'min':int(counts.min()),'max':int(counts.max())}
    # Descriptive surface forms, not a semantic/manual question-type taxonomy.
    prefixes={'yes_no':r'^(?:is|are|can|could|does|do|did|has|have|had|will|would|should|may|must)\b',
              'who':r'^who\b','what':r'^what\b','how':r'^how\b','when':r'^when\b','which':r'^which\b',
              'where':r'^where\b','why':r'^why\b'}
    assigned=pd.Series('other',index=full.index)
    for label,pattern in prefixes.items():
        assigned.loc[full.question_eng.str.strip().str.lower().str.match(pattern)]=label
    dataset['english_question_surface_forms']={k:int(v) for k,v in assigned.value_counts().items()}
    bleu=BLEU(tokenize='13a')
    chrf=CHRF(char_order=6,word_order=0,beta=2,lowercase=False,whitespace=False)
    f1s, scores = {}, {}
    for model,(label,human_file) in MODELS.items():
        response_path=ROOT/'data/model_qa_responses'/f'{model}-responses.csv'
        human_path=ROOT/'data/human_evaluations_qa'/human_file
        record(response_path);record(human_path)
        response=pd.read_csv(response_path)
        if response.id.duplicated().any():
            raise ValueError(f'Duplicate response IDs: {model}')
        response=response.set_index('id').sort_index()
        if response.index.tolist()!=list(range(len(val))):
            raise ValueError(f'Missing/extra response IDs: {model}')
        reference=val.answer_urdu.tolist()
        if response.answers.tolist()!=reference:
            raise ValueError(f'References differ from canonical validation data: {model}')
        question=next((c for c in response if c in ['question','questions']),None)
        if question is not None and response[question].tolist()!=val.question_urdu.tolist():
            raise ValueError(f'Questions differ from validation data: {model}')
        pred_col=next(c for c in response if c.startswith('prediction_text'))
        prediction=response[pred_col].tolist()
        if any(not isinstance(p,str) or not p.strip() for p in prediction):
            raise ValueError(f'Blank or non-string prediction: {model}')
        human=annotation_frame(human_path)
        if human.index.tolist()!=response.index.tolist():
            raise ValueError(f'Human annotation IDs differ: {model}')
        hp=next(c for c in human if c.startswith('prediction_text'))
        human_references=human.answers.tolist()
        if model == 'mt5-large':
            human_references=[ast.literal_eval(r)[0]['text'] for r in human_references]
        same_predictions = [annotation_text(p) for p in human[hp]]==[annotation_text(p) for p in prediction]
        if (not same_predictions and model != 'mt5-large') or [annotation_text(r) for r in human_references]!=[annotation_text(r) for r in reference]:
            raise ValueError(f'Human annotation text differs: {model}')
        f1=np.array([token_f1(p,r) for p,r in zip(prediction,reference)])
        uf1=np.array([token_f1(p,r,urdu_normalize) for p,r in zip(prediction,reference)])
        meteor=np.array([single_meteor_score(nltk.word_tokenize(r),nltk.word_tokenize(p)) for p,r in zip(prediction,reference)])
        row={'label':label,'n':len(response),'f1':float(100*f1.mean()),
             'exact_match':float(100*np.mean([squad_normalize(p)==squad_normalize(r) for p,r in zip(prediction,reference)])),
             'meteor':float(meteor.mean()),'sacrebleu':float(bleu.corpus_score(prediction,[reference]).score),
             'chrf':float(chrf.corpus_score(prediction,[reference]).score),'urdu_normalized_f1':float(100*uf1.mean()),
             'prediction_words_mean':float(np.mean([len(p.split()) for p in prediction])),
             'human_score_percent':float(100*human.Score.mean()),
             'human_counts':{str(v):int(human.Score.eq(v).sum()) for v in [0,.5,1]},
             'full_credit_percent':float(100*human.Score.eq(1).mean()),
             'strict_percent':float(100*human.Strict.mean()) if 'Strict' in human and human.Strict.notna().any() else None,
             'strict_counts':{str(v):int(human.Strict.eq(v).sum()) for v in [0,.5,1]} if 'Strict' in human and human.Strict.notna().any() else None,
             'human_text_alignment':'same by ID after whitespace/list-marker normalization for archived spreadsheet layout',
             'human_same_predictions':same_predictions,
             'recorded_metadata':{c:response[c].dropna().unique().tolist() for c in ['requested_model','model_version','reasoning_mode'] if c in response}}
        row['per_example']=[{'id':i,'context_index':int(val.context_index.iloc[i]),'f1':float(f),'urdu_normalized_f1':float(u),'human_score':float(h) if same_predictions else None} for i,(f,u,h) in enumerate(zip(f1,uf1,human.Score))]
        seen=val.context_index.isin(train.context_index).to_numpy()
        row['article_strata']={}
        for name,mask in [('seen',seen),('unseen',~seen)]:
            row['article_strata'][name]={'n':int(mask.sum()),'f1':float(100*f1[mask].mean()),
                'urdu_normalized_f1':float(100*uf1[mask].mean()),
                'prediction_words_mean':float(np.mean([len(p.split()) for p,keep in zip(prediction,mask) if keep]))}
        if not same_predictions:
            annotated=human[hp].tolist()
            row['human_text_alignment']='different mT5 output variant: 110 predictions differ; reference strings decoded from serialized SQuAD records'
            row['human_prediction_variant']={
                'different_prediction_count':sum(annotation_text(p)!=annotation_text(q) for p,q in zip(prediction,annotated)),
                'f1':float(100*np.mean([token_f1(p,r) for p,r in zip(annotated,reference)])),
                'meteor':float(np.mean([single_meteor_score(nltk.word_tokenize(r),nltk.word_tokenize(p)) for p,r in zip(annotated,reference)])),
                'sacrebleu':float(bleu.corpus_score(annotated,[reference]).score),
                'prediction_words_mean':float(np.mean([len(p.split()) for p in annotated])),
                'human_score_percent':row['human_score_percent']}
            for key in ['human_score_percent','human_counts','full_credit_percent','strict_percent','strict_counts']:
                row['human_prediction_variant'][key]=row[key]
                row[key]=None
        results['models'][model]=row
        if model in UPDATED:
            f1s[model]=f1;scores[model]=human.Score.to_numpy()
    results['metric_signatures']={'bleu':str(bleu.get_signature()),'chrf':str(chrf.get_signature()),
         'meteor':'NLTK single_meteor_score alpha=0.9,beta=3,gamma=0.5; word_tokenize default English; Porter stemmer/English WordNet defaults (not Urdu-aware)',
         'urdu_normalization':'NFC; Arabic kaf/yeh/alef-maksura -> Urdu kaf/yeh; remove combining marks, tatweel, Unicode Cf; punctuation -> spaces; decimal digits -> ASCII; lower and collapse whitespace; no stemming'}
    articles=val.context_index.to_numpy()
    groups=np.unique(articles)
    group_counts=np.array([(articles==a).sum() for a in groups])
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(groups),size=(bootstrap_samples,len(groups)))
    denominator=group_counts[draws].sum(axis=1)
    samples={}
    for key,vectors in [('f1',f1s),('human',scores)]:
        samples[key]={}
        for model,values in vectors.items():
            sums=np.array([values[articles==a].sum() for a in groups])
            samples[key][model]=100*sums[draws].sum(axis=1)/denominator
            results['models'][model][key+'_ci95']=np.percentile(samples[key][model],[2.5,97.5]).tolist()
    results['paired_comparisons']=[]
    for i,a in enumerate(UPDATED):
        for b in UPDATED[i+1:]:
            row={'a':a,'b':b}
            for key,vectors in [('f1',f1s),('human',scores)]:
                row[key+'_difference_pp']=float(100*np.mean(vectors[a]-vectors[b]))
                row[key+'_ci95']=np.percentile(samples[key][a]-samples[key][b],[2.5,97.5]).tolist()
            results['paired_comparisons'].append(row)
    # Judge analysis uses archived judgments and refreshed human labels; never API calls.
    jev_dir=ROOT/'data/automated_evaluations_qa/jev'
    frames=[]; audits=[]
    for path in sorted(jev_dir.glob('*_jev_verdicts.csv')):
        record(path)
        frame=pd.read_csv(path)
        model=path.stem.removesuffix('_jev_verdicts')
        key={'gemini-flash':'gemini-1.5-flash-002','gemini-pro':'gemini-1.5-pro-002'}.get(model,model)
        canonical=annotation_frame(ROOT/'data/human_evaluations_qa'/MODELS[key][1])
        if frame.set_index('id').sort_index().human_score.tolist()!=canonical.Score.tolist():
            raise ValueError(f'Stale copied JEV labels: {model}')
        hb=frame.human_score.ge(.5).to_numpy();jb=frame.noul.ge(.5).to_numpy()
        for _,r in frame.loc[hb!=jb].iterrows():
            audits.append({'model':model,'id':int(r.id),'context_index':int(val.context_index.iloc[int(r.id)]),
                          'human_score':float(r.human_score),'jev_probability':float(r.noul),
                          'question_english':val.question_eng.iloc[int(r.id)],'question_urdu':r.question,
                          'reference_urdu':r.ground_truth,'prediction_urdu':r.prediction,
                          'status':'audit candidate; no automatic annotation change'})
        frames.append(frame)
    pooled=pd.concat(frames,ignore_index=True)
    hb=pooled.human_score.ge(.5).to_numpy();jb=pooled.noul.ge(.5).to_numpy()
    from sklearn.metrics import cohen_kappa_score
    tp=int((hb&jb).sum());tn=int((~hb&~jb).sum());fp=int((~hb&jb).sum());fn=int((hb&~jb).sum())
    results['jev']={'n':len(pooled),'questions':len(val),'threshold':.5,'agreement':float((hb==jb).mean()),
         'kappa':float(cohen_kappa_score(hb,jb)),'always_positive_agreement':float(hb.mean()),
         'tp':tp,'tn':tn,'fp':fp,'fn':fn,'positive_recall':tp/(tp+fn),'specificity':tn/(tn+fp),
         'audit_candidates':len(audits),'returned_models':pooled.model.unique().tolist()}
    (out/'analysis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    (out/'annotation_audit_candidates.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'dataset':dataset,'models':{m:{k:v for k,v in r.items() if k!='per_example'} for m,r in results['models'].items()},'jev':results['jev']},ensure_ascii=False,indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'output/evaluation')
    parser.add_argument('--seed',type=int,default=25)
    parser.add_argument('--bootstrap-samples',type=int,default=10000)
    args=parser.parse_args()
    analyze(args.out,args.seed,args.bootstrap_samples)


if __name__=='__main__':
    main()
