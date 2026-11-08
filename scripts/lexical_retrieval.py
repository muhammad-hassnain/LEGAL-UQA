"""Small, deterministic BM25 baseline over the saved constitutional corpus.

No training, parameter selection, stemming, stopword list or inference service.
Uses positive smoothed IDF, k1=1.2, b=0.75; query terms contribute once each.
"""
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, documents, normalize, k1=1.2, b=.75):
        self.normalize, self.k1, self.b = normalize, k1, b
        self.documents = [Counter(normalize(d).split()) for d in documents]
        self.lengths = np.array([sum(d.values()) for d in self.documents])
        self.average_length = float(self.lengths.mean())
        if self.average_length == 0:
            raise ValueError('Empty retrieval corpus')
        df = Counter(term for d in self.documents for term in d)
        self.idf = {term: np.log1p((len(documents)-n+.5)/(n+.5)) for term,n in df.items()}

    def scores(self, question):
        scores = np.zeros(len(self.documents))
        norm = self.k1 * (1-self.b+self.b*self.lengths/self.average_length)
        for term in sorted(set(self.normalize(question).split())):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term,0) for d in self.documents])
            scores += self.idf[term] * tf*(self.k1+1)/(tf+norm)
        return scores


def evaluate(full, validation, normalize):
    # Both language corpora have 305 unique texts and one-to-one context IDs.
    docs=full.sort_values('context_index').drop_duplicates('context_index')
    if docs.context_urdu.duplicated().any() or docs.context_eng.duplicated().any():
        raise ValueError('Context-ID corpus differs from archived text-deduplicated corpus')
    identifiers=docs.context_index.to_numpy()
    results={}
    for language in ['eng','urdu']:
        retriever=BM25(docs['context_'+language].tolist(),normalize)
        records=[]
        for identifier,row in validation.iterrows():
            scores=retriever.scores(row['question_'+language])
            # Candidate order is ascending context ID; stable tie break has no query/target input.
            order=np.argsort(-scores,kind='stable')
            rank=int(np.flatnonzero(identifiers[order]==row.context_index)[0]+1)
            records.append({'id':int(identifier),'context_index':int(row.context_index),
                            'rank':rank,'ranked_context_ids':identifiers[order].tolist(),
                            'all_scores_zero':bool(np.all(scores==0)),
                            'top_score_tie_count':int(np.sum(scores==scores.max()))})
        ranks=np.array([r['rank'] for r in records])
        results[language]={'candidate_count':len(docs),'n':len(validation),
            'recall_percent':{k:float(100*np.mean(ranks<=k)) for k in [1,3,5]},
            'mrr':float(np.mean(1/ranks)),'all_zero_queries':sum(r['all_scores_zero'] for r in records),
            'top_tie_queries':sum(r['top_score_tie_count']>1 for r in records),'per_example':records}
    return {'status':'new offline lexical experiment', 'k1':1.2,'b':.75,
            'idf':'log(1+(N-df+0.5)/(df+0.5))', 'query_term_frequency':'binary',
            'tokenizer':'shared urdu_normalize then whitespace; no stemming/stopword removal',
            'tie_break':'ascending context ID', 'languages':results}
