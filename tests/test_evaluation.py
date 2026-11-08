"""Regression checks for consequential scoring and checkpoint behavior."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_saved import token_f1, urdu_normalize, annotation_frame
from scripts.jev.agreement import safe_kappa
from scripts.jev.pipeline import _load_checkpoint, input_fingerprint
from scripts.lexical_retrieval import BM25


class ScoringTests(unittest.TestCase):
    def test_lexical_retrieval_ranks_matches_and_handles_unseen_terms(self):
        retriever=BM25(['عدالت جج','پارلیمنٹ رکن','عدالت جج ملازمین'],urdu_normalize)
        scores=retriever.scores('پارلیمنٹ')
        self.assertEqual(int(np.argmax(scores)),1)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((scores>=0).all())
        self.assertTrue((retriever.scores('غیرموجود')==0).all())

    def test_repeated_tokens_count_as_multisets(self):
        self.assertAlmostEqual(token_f1('judge judge staff','judge staff'),.8)

    def test_completeness_and_negation_are_not_normalized_away(self):
        self.assertLess(token_f1('گورنر جج','گورنر جج ملازمین',urdu_normalize),1)
        self.assertLess(token_f1('رضامندی ضروری ہے','رضامندی ضروری نہیں ہے',urdu_normalize),1)

    def test_urdu_orthographic_variants_and_numbers(self):
        self.assertEqual(urdu_normalize('عَدالتِ، كے ۱۳ يہ'),urdu_normalize('عدالت کے 13 یہ'))

    def test_empty_answers_follow_squad_convention(self):
        self.assertEqual(token_f1('',''),1)
        self.assertEqual(token_f1('answer',''),0)

    def test_constant_prediction_kappa_is_defined_except_degenerate_case(self):
        self.assertEqual(safe_kappa([0,1,1],[1,1,1]),0)
        self.assertTrue(np.isnan(safe_kappa([1,1],[1,1])))
        self.assertEqual(safe_kappa([0,1],[0,1]),1)

    def test_duplicate_annotation_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.csv'
            pd.DataFrame({'id':[0,0],'Score':[1,0]}).to_csv(path,index=False)
            with self.assertRaises(ValueError):annotation_frame(path)

    def test_checkpoint_cannot_reuse_changed_answer_or_unversioned_run(self):
        row=pd.Series({'id':0,'question':'Q','ground_truth':'reference','prediction':'answer'})
        original=input_fingerprint(row)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.jsonl'
            path.write_text(json.dumps({'id':0,'input_sha256':original})+'\n')
            self.assertEqual(_load_checkpoint(path,{0:original}),{0})
            row['prediction']='different answer'
            with self.assertRaises(ValueError):_load_checkpoint(path,{0:input_fingerprint(row)})
            path.write_text(json.dumps({'id':0})+'\n')
            with self.assertRaises(ValueError):_load_checkpoint(path,{0:original})


if __name__=='__main__':unittest.main()
