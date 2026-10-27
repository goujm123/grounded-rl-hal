"""A frozen review quarantines question variants without rewriting source truth."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import content_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reasoning_review', ROOT / 'scripts/mcts/apply_reasoning_review.py')
review_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_module)


class ReasoningReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.audit_path = self.root / 'audit.json'
        self.output = self.root / 'curated'
        self.entries = [self.entry('e1', 'q1', 'A red square is visible.'),
                        self.entry('e2', 'q1', 'The visible square is red.'),
                        self.entry('e3', 'q2', 'No dog is visible.')]
        self.write_source()
        self.first_pass = self.root / 'first_pass.json'
        self.first_pass.write_text(json.dumps({'immutable': 'first pass'}))
        self.audit = {'visual_review_complete': True, 'inputs': {},
                      'first_pass_binding': {'path': str(self.first_pass), 'sha256': file_sha256(self.first_pass),
                                             'mutated_after_critic_join': False},
                      'entries': [self.audited(self.entries[0], support='ambiguous'), self.audited(self.entries[2])]}
        self.refresh_audit_bindings()

    @staticmethod
    def entry(entry_id, sample_id, observation):
        chain = '<think>' + observation + '</think><answer>yes</answer>'
        checksum = hashlib.sha256(chain.encode()).hexdigest()
        return {'id': entry_id, 'metadata': {'sample_key': sample_id, 'split': 'train', 'data_split': 'train',
                    'quality_status': 'accepted_by_automated_visual_verifier',
                    'quality_filter': {'candidate_sha256': checksum, 'decision_sha256': 'signed-source-decision'}},
                'messages': [{'role': 'user', 'content': 'Question?'}, {'role': 'assistant', 'content': chain}],
                'images': ['source.png'], 'gt_answer': 'yes'}

    @staticmethod
    def audited(entry, **changes):
        chain = entry['messages'][1]['content']
        review = {'support': 'strong', 'main_answer_support': 'supported', 'localization': 'not_applicable',
                  'redundant_evidence': False, 'source_label_ambiguity': None}
        review.update(changes)
        return {'sample_id': entry['metadata']['sample_key'], 'sft_entry_id': entry['id'],
                'chain_sha256': hashlib.sha256(chain.encode()).hexdigest(), 'exact_sft_chain': chain,
                'image': 'source.png', 'review': review}

    def write_source(self, val=None):
        for split, value in [('train', self.entries), ('val', val or []), ('test', [])]:
            (self.source / f'reasoning_chains_{split}.json').write_text(json.dumps(value) + '\n')
        (self.source / 'quality_decisions.jsonl').write_text('signed source provenance\n')
        manifest = {'all_parent_candidates_covered_once': True,
                    'outputs': {p.name: file_sha256(p) for p in self.source.iterdir() if p.name != 'quality_manifest.json'},
                    'verifier_policy_id': 'unchanged-critic-policy'}
        manifest['manifest_id'] = content_hash(manifest)
        (self.source / 'quality_manifest.json').write_text(json.dumps(manifest))

    def write_audit(self):
        self.audit.pop('visual_audit_id', None)
        self.audit['visual_audit_id'] = content_hash(self.audit)
        self.audit_path.write_text(json.dumps(self.audit))

    def refresh_audit_bindings(self):
        self.audit['inputs'] = {key: {'path': str(self.source / name), 'sha256': file_sha256(self.source / name)}
                               for key, name in [('sft_file', 'reasoning_chains_train.json'), ('sft_val', 'reasoning_chains_val.json'),
                                                 ('sft_test', 'reasoning_chains_test.json'), ('quality_manifest', 'quality_manifest.json')]}
        self.write_audit()

    def apply(self):
        return review_module.apply_review(self.source, self.audit_path, self.output)

    def test_quarantines_all_variants_and_preserves_original_quality_provenance(self):
        before = {str(p): file_sha256(p) for p in self.source.iterdir()}
        result = self.apply()
        self.assertEqual(result['input_chains'], 3)
        self.assertEqual(result['flagged_questions'], 1)
        self.assertEqual(result['excluded_chains'], 2)
        self.assertEqual(result['retained_questions'], 1)
        kept = json.loads((self.output / 'reasoning_chains_train.json').read_text())
        self.assertEqual([e['id'] for e in kept], ['e3'])
        original = deepcopy(kept[0])
        curation = original['metadata'].pop('review_curation')
        self.assertEqual(original, self.entries[2])
        self.assertFalse(curation['visual_correctness_certified'])
        for split in ['val', 'test']:
            self.assertEqual(json.loads((self.output / f'reasoning_chains_{split}.json').read_text()), [])
        ledger = [json.loads(line) for line in (self.output / 'review_curation_decisions.jsonl').read_text().splitlines()]
        self.assertFalse(ledger[1]['exact_chain_was_reviewed'])
        self.assertFalse(ledger[1]['retained'])
        self.assertEqual(before, {str(p): file_sha256(p) for p in self.source.iterdir()})
        for name, checksum in result['outputs'].items():
            self.assertEqual(file_sha256(self.output / name), checksum)

    def test_changed_source_export_is_rejected_even_with_refreshed_quality_manifest(self):
        self.entries[0]['gt_answer'] = 'no'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Audit bound source checksum'):
            self.apply()
        self.assertFalse(self.output.exists())

    def test_unknown_exact_chain_is_rejected(self):
        self.audit['entries'][0]['chain_sha256'] = 'f' * 64
        self.write_audit()
        with self.assertRaisesRegex(ValueError, 'unknown source question or exact chain'):
            self.apply()

    def test_wrong_source_entry_id_cannot_rebind_a_known_chain(self):
        self.audit['entries'][0]['sft_entry_id'] = 'different'
        self.write_audit()
        with self.assertRaisesRegex(ValueError, 'SFT identity differs'):
            self.apply()

    def test_first_pass_cannot_change_after_freeze(self):
        self.first_pass.write_text('changed first pass')
        with self.assertRaisesRegex(ValueError, 'Audit bound source checksum'):
            self.apply()

    def test_nonempty_heldout_exports_are_rejected(self):
        self.write_source(val=[self.entries[0]])
        self.refresh_audit_bindings()
        with self.assertRaisesRegex(ValueError, 'val/test must be empty'):
            self.apply()

    def test_heldout_entry_disguised_in_train_file_is_rejected(self):
        self.entries[0]['metadata']['data_split'] = 'test'
        self.write_source()
        self.refresh_audit_bindings()
        with self.assertRaisesRegex(ValueError, 'TRAIN entries only'):
            self.apply()

    def test_audit_must_be_complete_and_its_content_id_valid(self):
        self.audit['visual_review_complete'] = False
        self.write_audit()
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.apply()
        self.audit['visual_review_complete'] = True
        self.audit_path.write_text(json.dumps(self.audit))
        with self.assertRaisesRegex(ValueError, 'visual_audit_id'):
            self.apply()

    def test_every_nonclear_review_dimension_is_a_generic_question_flag(self):
        for changes, reason in [({'support': 'ambiguous'}, 'support:ambiguous'),
                                ({'localization': 'imprecise'}, 'localization:imprecise'),
                                ({'main_answer_support': 'ambiguous'}, 'main_answer_support:ambiguous'),
                                ({'source_label_ambiguity': 'uncertain wording'}, 'source_label_ambiguity'),
                                ({'redundant_evidence': True}, 'redundant_evidence')]:
            review = self.audited(self.entries[0], **changes)['review']
            self.assertIn(reason, review_module.review_reasons(review))
        self.assertEqual(review_module.review_reasons(self.audited(self.entries[0])['review']), [])

    def test_explicit_unsupported_claims_override_a_strong_summary(self):
        self.audit['entries'][0]['review']['support'] = 'strong'
        self.audit['entries'][0]['review']['unsupported_or_unverifiable_claims'] = ['The paddle is vertical.']
        self.write_audit()
        result = self.apply()
        self.assertEqual(result['flagged_question_reasons'], {'q1': ['unsupported_or_unverifiable_claims']})
        self.assertEqual(result['excluded_chains'], 2)
        self.assertEqual(result['source_tool_sha256'], file_sha256(ROOT / 'scripts/mcts/apply_reasoning_review.py'))
        for invalid in ['a claim', [12], None]:
            review = self.audited(self.entries[0])['review']
            review['unsupported_or_unverifiable_claims'] = invalid
            with self.assertRaisesRegex(ValueError, 'list of strings'):
                review_module.review_reasons(review)

    def test_existing_output_is_never_overwritten(self):
        self.apply()
        path = self.output / 'reasoning_chains_train.json'
        checksum = file_sha256(path)
        with self.assertRaisesRegex(ValueError, 'new directory'):
            self.apply()
        self.assertEqual(file_sha256(path), checksum)


if __name__ == '__main__':
    unittest.main()
