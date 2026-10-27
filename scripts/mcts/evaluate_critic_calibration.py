#!/usr/bin/env python3
"""Audit exact-chain critic calibration against a frozen separate visual review.

Validation requires a policy lock written before validation calls. Locks contain
fixture_sha256, filter_version, model_identity, verifier_config (without base_url),
verifier_fingerprints, locked_at_utc and policy_source_sha256 (path/hash mapping).
No reference labels are passed to inference and no decisions are rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import atomic_json, content_hash

ROOT = Path(__file__).resolve().parents[2]


def load_merger():
    spec = importlib.util.spec_from_file_location('calibration_record_audit', ROOT / 'scripts/mcts/merge_quality_shards.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_content(path, field):
    value = json.loads(Path(path).read_text())
    if value.get(field) != content_hash({k: v for k, v in value.items() if k != field}):
        raise ValueError(f'Invalid {field}: {path}')
    return value


def utc(value):
    if not isinstance(value, str):
        raise ValueError('Missing frozen/lock/record UTC timestamp')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('UTC timestamp must include its timezone')
    return result


def normalized_config(value):
    return {k: v for k, v in value.items() if k != 'base_url'}


def uncertainty_signals(value):
    result = []
    def walk(x, path):
        if isinstance(x, dict):
            for key, item in x.items():
                location = path + '.' + key
                if key in {'label', 'image_answer', 'clarity', 'status', 'precision',
                           'whole_observation_support', 'explanation_support', 'answer_support'} and item == 'uncertain':
                    result.append(location)
                if key in {'unambiguous', 'question_unambiguous'} and item is False:
                    result.append(location)
                walk(item, location)
        elif isinstance(x, list):
            for i, item in enumerate(x):
                walk(item, f'{path}[{i}]')
    walk(value.get('passes', {}), 'passes')
    result.extend(reason for reason in value['reasons'] if 'uncertain' in reason or 'ambiguous' in reason)
    return sorted(set(result))


def load_fixture(path, partition):
    fixture = checked_content(path, 'fixture_id')
    frozen_at = utc(fixture.get('frozen_at_utc', fixture.get('created_at_utc')))
    ids, groups, examples = set(), defaultdict(set), []
    for example in fixture['examples']:
        key = example['example_id']
        if key in ids:
            raise ValueError('Duplicate fixture example ID')
        ids.add(key)
        ci = example['critic_input']
        if hashlib.sha256(ci['candidate_chain'].encode()).hexdigest() != example['chain_sha256']:
            raise ValueError('Fixture exact-chain checksum changed')
        if ci.get('coordinate_frame') != 'original_pixels':
            raise ValueError('Calibration requires original-pixel chains')
        if example['automated_reference']['expected_decision'] not in {'pass', 'reject', 'uncertain'}:
            raise ValueError('Unknown separately reviewed reference decision')
        if example['partition'] not in {'calibration', 'validation'}:
            raise ValueError('Unknown reference partition')
        for identity in [('image_group', example['provenance']['image_group']),
                         ('image_sha256', ci['image_sha256']), ('image_path', ci['image'])]:
            groups[identity].add(example['partition'])
        if example['partition'] == partition:
            examples.append(example)
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError('Calibration and validation share an image group')
    if not examples:
        raise ValueError('Requested reference partition is empty')
    return fixture, examples, frozen_at


def validate_decision(row, example, summary, manifest, audit, merge):
    ci, verdict = example['critic_input'], row['verdict']
    if row.get('chain_sha256') != example['chain_sha256']:
        raise ValueError('Ledger exact-chain checksum differs from frozen reference')
    expected = {'candidate_sha256': example['chain_sha256'], 'image_sha256': ci['image_sha256'],
                'question_sha256': hashlib.sha256(ci['question'].encode()).hexdigest(), 'geometry_sha256': None,
                'source_fingerprints': {'critic_inputs_sha256': summary['critic_inputs_sha256']},
                'verifier_fingerprint': summary['verifier_fingerprint']}
    if verdict.get('binding') != expected:
        raise ValueError('Decision source/question/image/model binding changed')
    candidate = {'candidate_chain': ci['candidate_chain'], 'image_sha256': ci['image_sha256'],
                 'question': ci['question'], 'image_geometry': None,
                 'source_fingerprints': expected['source_fingerprints']}
    # Calibration has no frozen transport geometry, but all other source,
    # complete inventory, prompt, stage, decision and attempt checks are shared
    # with the production merger rather than maintaining a weaker test path.
    proxy = dict(row, accepted=verdict.get('accepted'), duplicate_exact_chain=False)
    merge.validate_verdict(proxy, candidate, manifest, audit)
    return verdict



def evaluate(fixture_path, ledger_paths, partition, policy_lock=None):
    fixture_path = Path(fixture_path).resolve()
    fixture, examples, frozen_at = load_fixture(fixture_path, partition)
    selected = {e['example_id']: e for e in examples}
    if partition == 'validation' and policy_lock is None:
        raise ValueError('Locked validation requires --policy-lock')
    lock = checked_content(policy_lock, 'lock_id') if policy_lock else None
    if lock:
        if lock['fixture_sha256'] != file_sha256(fixture_path):
            raise ValueError('Policy lock binds another reference fixture')
        for path, digest in lock['policy_source_sha256'].items():
            if not Path(path).is_file() or file_sha256(path) != digest:
                raise ValueError('Locked critic policy source changed')
        locked_at = utc(lock['locked_at_utc'])
        if locked_at < frozen_at:
            raise ValueError('Policy was locked before reference freeze')
    merge = load_merger()
    audit = merge.CriticRecordAudit()
    rows, bindings, summaries, policy, source_inputs = {}, {}, [], None, set()
    for ledger_path in ledger_paths:
        ledger_path = Path(ledger_path).resolve()
        if str(ledger_path) in bindings:
            raise ValueError('Duplicate ledger path')
        summary_path = ledger_path.with_name('calibration_summary.json')
        summary = checked_content(summary_path, 'summary_id')
        ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        if not ledger_rows or len(ledger_rows) != summary['examples']:
            raise ValueError('Completed calibration summary and ledger disagree')
        versions = {row['verdict']['version'] for row in ledger_rows}
        if len(versions) != 1:
            raise ValueError('Ledger mixes critic versions')
        manifest = {'filter_version': versions.pop(), 'verifier_config': summary['verifier_config'],
                    'model_identity': summary['model_identity'], 'verifier_fingerprint': summary['verifier_fingerprint']}
        current = {k: manifest[k] for k in ['filter_version', 'model_identity']}
        current['verifier_config'] = normalized_config(summary['verifier_config'])
        if policy is not None and policy != current:
            raise ValueError('Calibration ledgers mix policies, models, or inference settings')
        policy = current
        if lock:
            for key, value in policy.items():
                if lock[key] != value:
                    raise ValueError('Validation critic differs from locked policy')
            if summary['verifier_fingerprint'] not in lock['verifier_fingerprints']:
                raise ValueError('Validation critic fingerprint was not locked')
        if summary.get('reference_fixture_sha256') not in {None, file_sha256(fixture_path)}:
            raise ValueError('Ledger joined a different reference fixture')
        for row in ledger_rows:
            key = row.get('example_id')
            if key not in selected:
                raise ValueError('Ledger includes an unknown or unrequested-partition example')
            if key in rows:
                raise ValueError('Duplicate decision for a reference example')
            verdict = validate_decision(row, selected[key], summary, manifest, audit, merge)
            rows[key] = verdict
        audit.verify_fingerprint(manifest)
        source_inputs.add(summary['critic_inputs_sha256'])
        summaries.append(manifest)
        bindings[str(ledger_path)] = file_sha256(ledger_path)
        bindings[str(summary_path)] = file_sha256(summary_path)
    if set(rows) != set(selected):
        raise ValueError('Missing decisions; every selected reference must appear exactly once')
    # Every recorded call/cache must follow reference freeze and, for validation,
    # policy lock. Reusing calibration results as locked validation cannot pass.
    cutoff = max(frozen_at, locked_at) if lock else frozen_at
    for path in audit.files:
        timestamp = utc(json.loads(Path(path).read_text())['generated_at_utc'])
        if timestamp < cutoff:
            raise ValueError('Critic call/cache predates reference freeze or validation policy lock')
    counts = Counter({key: 0 for key in ['examples', 'false_rejections', 'false_accepts', 'protocol_error',
                                            'accepted', 'uncertainty_withheld', 'definite_or_policy_rejection',
                                            'uncertain_reference_withheld', 'expected_pass', 'expected_reject',
                                            'expected_uncertain', 'observed_pass', 'observed_reject', 'observed_error',
                                            'expected_pass_observed_pass', 'expected_reject_observed_reject']})
    per_cell, entries = defaultdict(Counter), []
    for example in examples:
        key, reference = example['example_id'], example['automated_reference']['expected_decision']
        verdict = rows[key]
        complete = verdict['verification_complete']
        actual = 'error' if not complete else 'pass' if verdict['accepted'] else 'reject'
        signals = uncertainty_signals(verdict) if complete else []
        category = 'protocol_error' if not complete else 'accepted' if verdict['accepted'] else 'uncertainty_withheld' if signals else 'definite_or_policy_rejection'
        increments = ['examples', 'expected_' + reference, 'observed_' + actual,
                      f'expected_{reference}_observed_{actual}', category]
        if reference == 'pass' and actual == 'reject':
            increments.append('false_rejections')
        if reference == 'reject' and actual == 'pass':
            increments.append('false_accepts')
        if reference == 'uncertain' and actual == 'reject':
            increments.append('uncertain_reference_withheld')
        if reference == 'reject' and actual == 'reject' and not signals:
            increments.append('reject_reference_definite_or_policy_rejection')
        for name in increments:
            counts[name] += 1
            per_cell[example['provenance']['cell']][name] += 1
        entries.append({'example_id': key, 'partition': partition, 'provenance': example['provenance'],
                        'chain_sha256': example['chain_sha256'], 'expected_decision': reference,
                        'observed_decision': actual, 'outcome_kind': category, 'uncertainty_signals': signals,
                        'reasons': verdict['reasons'], 'decision_sha256': verdict['decision_sha256']})
    rate = lambda numerator, denominator: counts[numerator] / counts[denominator] if counts[denominator] else None
    report = {'schema_version': 1, 'partition': partition, 'fixture_id': fixture['fixture_id'],
              'fixture': {'path': str(fixture_path), 'sha256': file_sha256(fixture_path)},
              'reference_type': 'separately_reviewed_automated_visual_reference_not_human_gold',
              'policy_lock': {'path': str(Path(policy_lock).resolve()), 'sha256': file_sha256(policy_lock)} if lock else None,
              'critic_policy': policy, 'counts': dict(counts),
              'supported_chain_acceptance': rate('expected_pass_observed_pass', 'expected_pass'),
              'bad_claim_rejection_rate': rate('expected_reject_observed_reject', 'expected_reject'),
              'false_rejection_rate': rate('false_rejections', 'expected_pass'),
              'false_accept_rate': rate('false_accepts', 'expected_reject'),
              'protocol_errors': counts['protocol_error'], 'complete_exact_coverage': True,
              'all_decisions_valid': counts['protocol_error'] == 0,
              'per_cell': {k: dict(v) for k, v in sorted(per_cell.items())}, 'examples': entries,
              'source_bindings': {**bindings, **audit.files},
              'limitations': ['Automated separate visual review is fallible and is not human ground truth.',
                              'Protocol errors earn no known-bad rejection credit and are not counted as semantic false rejections.',
                              'Uncertainty-driven withholding is reported separately from definite or policy rejection.',
                              'All rates use all references in that class, including protocol errors in the denominator.',
                              'Calibration and locked validation share no image groups; validation cannot be used to tune this policy.']}
    merge.checked_bindings(report['source_bindings'])
    if file_sha256(fixture_path) != report['fixture']['sha256']:
        raise ValueError('Reference fixture changed during evaluation')
    report['evaluation_id'] = content_hash(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, action='append', required=True)
    parser.add_argument('--partition', choices=['calibration', 'validation'], required=True)
    parser.add_argument('--policy-lock', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evaluation output must be new; preserve prior reports')
    report = evaluate(args.fixture, args.ledger, args.partition, args.policy_lock)
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in ['partition', 'counts', 'supported_chain_acceptance',
                                           'bad_claim_rejection_rate', 'protocol_errors', 'evaluation_id']}, indent=2))


if __name__ == '__main__':
    main()
