---
license: mit
language:
  - en
  - ur
task_categories:
  - question-answering
---

# LEGAL-UQA data statement

LEGAL-UQA contains 619 aligned English/Urdu-script questions, answers, and
contexts from 305 of 312 extracted constitutional chunks. Questions are
model-generated in a layperson style; answers are generated and translated.
They are not naturally collected consultations or independently authored legal
opinions. The Urdu register follows the OCR-derived, model-refined Constitution.

`complete_dataset` contains 495 training and 124 validation rows. The split is
by QA row (`test_size=0.2`, `random_state=25`), with 95 shared context IDs and
101 validation questions about training articles. An article chunk is an
English article entry separated by manual delimiters, generally including its
heading and clauses; some chunks contain only an omitted-provision notice.
`context_index` is the zero-based position of that chunk in the extracted list,
not a constitutional article number. Urdu passages were paired by row after OCR
and cleanup, without an independent audit of exact article boundaries. The
pandas index field is storage metadata and is not a semantic feature.

The seven semantic fields are `question_eng`, `question_urdu`, `answer_eng`,
`answer_urdu`, `context_eng`, `context_urdu`, and `context_index`. There are no
extractive span labels, unanswerability labels, or manually validated legal-topic
or reasoning-complexity labels. Statistics and source hashes can be generated with
`python -m scripts.evaluate_saved` in `output/evaluation/analysis.json`.

The author confirmed three Urdu-proficient human evaluators: two graded answers
and a third resolved disagreements. The primary rubric requires semantic equivalence
to the reference with no missing detail, and marks correct but partial answers
wrong. Individual ratings, blinding, and inter-annotator agreement are not
recorded. Logged corrections change Claude ID 14 from 1 to 0
and Gemma IDs 14/99 from .5 to 0.
The labels are recorded judgments rather than an independent expert legal
audit. mT5's annotated output variant differs from its response CSV on
110/124 answers; the evaluation keeps these variants separate.

The source covers state an amendment cutoff of 31 May 2018, with prefaces
dated 31 May 2018 (English) and 12 July 2018 (Urdu). Amendment alignment
and current legal validity have not been established. Scores are relative
to the saved contexts.

The authors designate MIT for project code and author-contributed dataset
material. Third-party constitutional documents, model assets, and metric implementations retain their own rights and licenses. Public access
to a source document is not a grant by this project of rights it does not own.

Use for research on constitutional reading comprehension. The data does not
establish suitability for legal advice, multi-hop legal reasoning, or end-to-end
retrieval-augmented answering. No private consultations were collected.
