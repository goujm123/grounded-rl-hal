"""Separate visual references must stay exact, disjoint and policy-locked."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import QualityVerifier, VerifierConfig, content_hash

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module('calibration_critic_fixtures', 'tests/test_quality_verifier.py')
evaluator = module('critic_calibration_evaluator', 'scripts/mcts/evaluate_critic_calibration.py')


class CalibrationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = self.root / 'fixture.json'
        self.examples = []
        self.ledgers = []
        for i, expected in enumerate(['pass', 'reject', 'reject', 'pass']):
            image = self.root / f'image_{i}.png'
            Image.new('RGB', (64 + i, 64), 'red').save(image)
            chain = '<think>A red square is visible.</think><answer>yes</answer>'
            self.examples.append({'example_id': f'opaque_{i}', 'chain_sha256': evaluator.hashlib.sha256(chain.encode()).hexdigest(),
                                  'critic_input': {'image': str(image), 'image_sha256': file_sha256(image),
                                                   'original_size': [64 + i, 64], 'coordinate_frame': 'original_pixels',
                                                   'question': 'Is there a red square?', 'candidate_chain': chain},
                                  'automated_reference': {'expected_decision': expected},
                                  'partition': 'calibration' if i < 3 else 'validation',
                                  'provenance': {'image_group': f'group_{i}', 'sample_id': f'train_{i}', 'cell': 'state:yes'}})
        self.write_fixture()

    def write_fixture(self):
        value = {'schema_version': 1, 'created_at_utc': '2000-01-01T00:00:00+00:00', 'examples': self.examples}
        value['fixture_id'] = content_hash(value)
        self.fixture.write_text(json.dumps(value))

    def generate(self, indexes=(0, 1, 2), mixed_model=False, changes_override=None):
        for i in indexes:
            record = self.examples[i]
            inputs = self.root / f'inputs_{i}.jsonl'
            inputs.write_text(json.dumps({k: record[k] for k in ['example_id', 'chain_sha256', 'critic_input']}) + '\n')
            def unsupported(v):
                v['claims'][0]['status'] = 'unsupported'
                return v
            changes = {'scene_claims': unsupported} if i == 1 else {'scene_claims': lambda v: '{'} if i == 2 else None
            if changes_override and i in changes_override:
                changes = changes_override[i]
            config = VerifierConfig(base_url=f'http://localhost:{9011 + i}/v1', reject_repetition=False)
            verifier = QualityVerifier(config, self.root / f'cache_{i}', client=fixtures.FakeCritic(changes),
                                       model_identity={'revision': 'different' if mixed_model and i == 1 else 'test'}, verify_endpoint=False)
            output = self.root / f'output_{i}'
            with contextlib.redirect_stdout(io.StringIO()):
                fixtures.quality_filter.verify_inputs(inputs, output, self.root, verifier)
            self.ledgers.append(output / 'quality_decisions.jsonl')

    def lock(self, after_calls=False):
        summary = json.loads(self.ledgers[0].with_name('calibration_summary.json').read_text())
        row = json.loads(self.ledgers[0].read_text().splitlines()[0])
        source = self.root / 'policy.py'
        source.write_text('frozen policy')
        value = {'fixture_sha256': file_sha256(self.fixture), 'filter_version': row['verdict']['version'],
                 'model_identity': summary['model_identity'],
                 'verifier_config': evaluator.normalized_config(summary['verifier_config']),
                 'verifier_fingerprints': [summary['verifier_fingerprint']],
                 'locked_at_utc': '2999-01-01T00:00:00+00:00' if after_calls else '2001-01-01T00:00:00+00:00',
                 'policy_source_sha256': {str(source): file_sha256(source)}}
        value['lock_id'] = content_hash(value)
        path = self.root / 'policy_lock.json'
        path.write_text(json.dumps(value))
        return path

    def test_protocol_failure_gets_no_bad_claim_rejection_credit(self):
        self.generate()
        result = evaluator.evaluate(self.fixture, self.ledgers, 'calibration')
        self.assertEqual(result['counts']['expected_pass_observed_pass'], 1)
        self.assertEqual(result['counts']['expected_reject_observed_reject'], 1)
        self.assertEqual(result['counts']['expected_reject_observed_error'], 1)
        self.assertEqual(result['bad_claim_rejection_rate'], .5)
        self.assertEqual(result['protocol_errors'], 1)
        self.assertEqual(result['counts']['false_rejections'], 0)
        self.assertTrue(result['complete_exact_coverage'])
        self.assertFalse(result['all_decisions_valid'])
        self.assertGreater(len(result['source_bindings']), len(self.ledgers))

    def test_supported_rejection_and_bad_chain_acceptance_are_counted_separately(self):
        def uncertain(v):
            v['claims'][0]['status'] = 'uncertain'
            return v
        self.generate(changes_override={0: {'scene_claims': uncertain}, 1: None})
        result = evaluator.evaluate(self.fixture, self.ledgers, 'calibration')
        self.assertEqual(result['counts']['false_rejections'], 1)
        self.assertEqual(result['counts']['false_accepts'], 1)
        self.assertEqual(result['counts']['uncertainty_withheld'], 1)
        self.assertEqual(result['false_rejection_rate'], 1)
        self.assertEqual(result['false_accept_rate'], .5)
        self.assertEqual(result['protocol_errors'], 1)

    def test_ambiguous_reference_withholding_is_not_a_false_rejection(self):
        self.examples[0]['automated_reference']['expected_decision'] = 'uncertain'
        self.write_fixture()
        def uncertain(v):
            v['claims'][0]['status'] = 'uncertain'
            return v
        self.generate(changes_override={0: {'scene_claims': uncertain}})
        result = evaluator.evaluate(self.fixture, self.ledgers, 'calibration')
        self.assertEqual(result['counts']['uncertain_reference_withheld'], 1)
        self.assertEqual(result['counts']['false_rejections'], 0)
        self.assertIsNone(result['supported_chain_acceptance'])

    def test_missing_duplicate_and_unrequested_partition_are_rejected(self):
        self.generate()
        with self.assertRaisesRegex(ValueError, 'Missing decisions'):
            evaluator.evaluate(self.fixture, self.ledgers[:-1], 'calibration')
        with self.assertRaisesRegex(ValueError, 'Duplicate ledger'):
            evaluator.evaluate(self.fixture, self.ledgers + self.ledgers[:1], 'calibration')
        self.generate(indexes=(3,))
        with self.assertRaisesRegex(ValueError, 'unrequested-partition'):
            evaluator.evaluate(self.fixture, self.ledgers, 'calibration')

    def test_changed_chain_or_signed_decision_cannot_be_scored(self):
        self.generate()
        ledger = self.ledgers[0]
        row = json.loads(ledger.read_text())
        changed = deepcopy(row)
        changed['chain_sha256'] = '0' * 64
        ledger.write_text(json.dumps(changed) + '\n')
        with self.assertRaisesRegex(ValueError, 'exact-chain checksum'):
            evaluator.evaluate(self.fixture, self.ledgers, 'calibration')
        changed = deepcopy(row)
        changed['verdict']['accepted'] = False
        ledger.write_text(json.dumps(changed) + '\n')
        with self.assertRaisesRegex(ValueError, 'decision SHA256'):
            evaluator.evaluate(self.fixture, self.ledgers, 'calibration')

    def test_image_groups_cannot_cross_calibration_and_validation(self):
        self.examples[3]['provenance']['image_group'] = self.examples[0]['provenance']['image_group']
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, 'share an image group'):
            evaluator.evaluate(self.fixture, [], 'calibration')

    def test_mixed_model_ledgers_cannot_be_compared_as_one_policy(self):
        self.generate(mixed_model=True)
        with self.assertRaisesRegex(ValueError, 'mix policies, models'):
            evaluator.evaluate(self.fixture, self.ledgers, 'calibration')

    def test_validation_requires_lock_and_cannot_reuse_prelock_calls(self):
        self.generate(indexes=(3,))
        with self.assertRaisesRegex(ValueError, 'requires --policy-lock'):
            evaluator.evaluate(self.fixture, self.ledgers, 'validation')
        lock = self.lock()
        result = evaluator.evaluate(self.fixture, self.ledgers, 'validation', lock)
        self.assertEqual(result['supported_chain_acceptance'], 1)
        lock = self.lock(after_calls=True)
        with self.assertRaisesRegex(ValueError, 'predates reference freeze'):
            evaluator.evaluate(self.fixture, self.ledgers, 'validation', lock)

    def test_bound_cache_tampering_is_detected(self):
        self.generate()
        row = json.loads(self.ledgers[0].read_text())
        path = Path(row['verdict']['cache_records'][0]['cache_file'])
        value = json.loads(path.read_text())
        value['parsed_response']['label'] = 'no'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'record_sha256'):
            evaluator.evaluate(self.fixture, self.ledgers, 'calibration')


if __name__ == '__main__':
    unittest.main()
