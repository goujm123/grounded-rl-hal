#!/usr/bin/env python3
"""Export a new TRAIN view quarantining questions flagged by a frozen visual audit.

The automated critic policy, source exports, source labels and images are never
changed. All accepted variants of a flagged question are withheld, including
variants absent from the reviewed sample. Unreviewed retained chains are not
certified as correct by this curation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from vlmsearch.data_manifest import file_sha256
from vlmsearch.quality_verifier import content_hash

VERSION = 'question_quarantine_from_frozen_visual_review_v1'
SPLITS = ('train', 'val', 'test')


def checked_json(path, field=None):
    value = json.loads(Path(path).read_text())
    if field and value.get(field) != content_hash({k: v for k, v in value.items() if k != field}):
        raise ValueError(f'Invalid {field}: {path}')
    return value


def bound_file(binding, expected_path=None):
    path = Path(binding['path']).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError('Audit source path differs from supplied exports')
    if not path.is_file() or file_sha256(path) != binding['sha256']:
        raise ValueError('Audit bound source checksum changed')
    return path


def chain_identity(entry):
    metadata = entry.get('metadata', {})
    if metadata.get('data_split', metadata.get('split')) != 'train' or metadata.get('split', 'train') != 'train':
        raise ValueError('Review curation admits TRAIN entries only')
    sample_id = metadata.get('sample_key')
    if not isinstance(sample_id, str) or not sample_id or not isinstance(entry.get('id'), str):
        raise ValueError('Source entry lacks its sample or SFT identity')
    assistants = [m['content'] for m in entry.get('messages', []) if m.get('role') == 'assistant']
    if len(assistants) != 1 or not isinstance(assistants[0], str):
        raise ValueError('Source entry must contain one complete assistant chain')
    chain = assistants[0]
    import hashlib
    checksum = hashlib.sha256(chain.encode()).hexdigest()
    if metadata.get('quality_status') != 'accepted_by_automated_visual_verifier':
        raise ValueError('Source entry was not accepted by the automated quality filter')
    quality = metadata.get('quality_filter', {})
    if quality.get('candidate_sha256') != checksum or not quality.get('decision_sha256'):
        raise ValueError('Original quality provenance does not bind the exact source chain')
    return sample_id, checksum, chain


def review_reasons(review):
    allowed = {'support': {'strong', 'ambiguous', 'unsupported'},
               'main_answer_support': {'supported', 'ambiguous', 'unsupported'},
               'localization': {'supported', 'not_applicable', 'imprecise', 'wrong', 'uncertain'}}
    for key, values in allowed.items():
        if review.get(key) not in values:
            raise ValueError(f'Invalid or missing visual review {key}')
    if not isinstance(review.get('redundant_evidence'), bool) or 'source_label_ambiguity' not in review:
        raise ValueError('Visual review lacks repetition or source-label ambiguity status')
    reasons = []
    if review['support'] != 'strong':
        reasons.append('support:' + review['support'])
    if review['main_answer_support'] != 'supported':
        reasons.append('main_answer_support:' + review['main_answer_support'])
    if review['localization'] not in {'supported', 'not_applicable'}:
        reasons.append('localization:' + review['localization'])
    if review['source_label_ambiguity'] is not None:
        reasons.append('source_label_ambiguity')
    if review['redundant_evidence']:
        reasons.append('redundant_evidence')
    unsupported = review.get('unsupported_or_unverifiable_claims', [])
    if not isinstance(unsupported, list) or any(not isinstance(item, str) for item in unsupported):
        raise ValueError('Unsupported/unverifiable claims must be a list of strings')
    if unsupported:
        reasons.append('unsupported_or_unverifiable_claims')
    return reasons


def apply_review(input_dir, audit_path, output_dir):
    input_dir, audit_path, output_dir = (Path(p).resolve() for p in (input_dir, audit_path, output_dir))
    if output_dir.exists() or output_dir == input_dir:
        raise ValueError('Review-curated output must be a new directory')
    source = {}
    exports = {}
    for split in SPLITS:
        path = input_dir / f'reasoning_chains_{split}.json'
        exports[split] = checked_json(path)
        if not isinstance(exports[split], list) or (split != 'train' and exports[split]):
            raise ValueError('Review curation admits TRAIN only; val/test must be empty')
        source[str(path)] = file_sha256(path)
    manifest_path = input_dir / 'quality_manifest.json'
    parent = checked_json(manifest_path, 'manifest_id')
    if not parent.get('all_parent_candidates_covered_once'):
        raise ValueError('Source quality filtering is not complete')
    required_outputs = {'quality_decisions.jsonl', *(f'reasoning_chains_{split}.json' for split in SPLITS)}
    if not required_outputs <= parent.get('outputs', {}).keys():
        raise ValueError('Source quality manifest lacks required output bindings')
    for name, checksum in parent['outputs'].items():
        if Path(name).name != name or file_sha256(input_dir / name) != checksum:
            raise ValueError('Completed source quality output checksum changed')
        source[str(input_dir / name)] = checksum
    source[str(manifest_path)] = file_sha256(manifest_path)
    audit = checked_json(audit_path)
    if audit.get('visual_review_complete') is not True:
        raise ValueError('Visual audit is not complete and frozen')
    id_field = next((key for key in ['visual_audit_id', 'audit_id'] if key in audit), None)
    if id_field:
        checked_json(audit_path, id_field)
    audit_id = audit[id_field] if id_field else content_hash(audit)
    source[str(audit_path)] = file_sha256(audit_path)
    bound_file(audit['inputs']['sft_file'], input_dir / 'reasoning_chains_train.json')
    for key, split in [('sft_val', 'val'), ('sft_test', 'test')]:
        if key in audit['inputs']:
            bound_file(audit['inputs'][key], input_dir / f'reasoning_chains_{split}.json')
    if 'quality_manifest' in audit['inputs']:
        bound_file(audit['inputs']['quality_manifest'], manifest_path)
    frozen = audit.get('first_pass_binding')
    if frozen:
        if frozen.get('mutated_after_critic_join') is not False:
            raise ValueError('First-pass review was changed after critic inspection')
        frozen_path = bound_file(frozen)
        source[str(frozen_path)] = frozen['sha256']
    entries = exports['train']
    identities, entry_ids = {}, set()
    for index, entry in enumerate(entries):
        sample_id, checksum, chain = chain_identity(entry)
        key = (sample_id, checksum)
        if key in identities or entry['id'] in entry_ids:
            raise ValueError('Source contains duplicate exact-chain or SFT identities')
        entry_ids.add(entry['id'])
        identities[key] = (index, entry, chain)
    if not entries:
        raise ValueError('Source quality-filtered training export is empty')
    flagged, reviewed, audits_by_question = defaultdict(set), set(), defaultdict(list)
    for row in audit['entries']:
        key = (row['sample_id'], row['chain_sha256'])
        if key not in identities:
            raise ValueError('Audit refers to an unknown source question or exact chain')
        index, entry, chain = identities[key]
        if row.get('exact_sft_chain') != chain or row.get('sft_entry_id') != entry['id']:
            raise ValueError('Audited exact chain or SFT identity differs from its source')
        if key in reviewed:
            raise ValueError('Audit repeats the same exact source chain')
        reviewed.add(key)
        if 'image' in row and row['image'] not in entry.get('images', []):
            raise ValueError('Audited image differs from source entry')
        reasons = review_reasons(row['review'])
        flagged[row['sample_id']].update(reasons)
        audits_by_question[row['sample_id']].append({'sft_entry_id': entry['id'], 'chain_sha256': row['chain_sha256'],
                                                    'review_sha256': content_hash(row['review']), 'reasons': reasons})
    if not reviewed:
        raise ValueError('Audit contains no exact reviewed source chains')
    flagged = {key: sorted(values) for key, values in flagged.items() if values}
    kept, decisions = [], []
    counts = Counter()
    for index, entry in enumerate(entries):
        sample_id, checksum, _ = chain_identity(entry)
        reasons = flagged.get(sample_id, [])
        decision = {'source_entry_index': index, 'sft_entry_id': entry['id'], 'sample_id': sample_id,
                    'chain_sha256': checksum, 'retained': not reasons, 'reasons': reasons,
                    'exact_chain_was_reviewed': (sample_id, checksum) in reviewed,
                    'question_review_records': audits_by_question.get(sample_id, [])}
        decisions.append(decision)
        counts['excluded_chains' if reasons else 'retained_chains'] += 1
        if not reasons:
            candidate = deepcopy(entry)
            candidate['metadata']['review_curation'] = {'version': VERSION, 'audit_id': audit_id,
                'audit_sha256': source[str(audit_path)], 'source_quality_manifest_id': parent['manifest_id'],
                'source_entry_index': index, 'review_ledger_entry_index': index,
                'exact_chain_was_reviewed': decision['exact_chain_was_reviewed'],
                'status': 'retained_after_flagged_question_quarantine', 'visual_correctness_certified': False}
            kept.append(candidate)
    original_questions = {chain_identity(e)[0] for e in entries}
    retained_questions = {chain_identity(e)[0] for e in kept}
    reason_questions = Counter(reason for reasons in flagged.values() for reason in reasons)
    reason_chains = Counter(reason for decision in decisions if not decision['retained'] for reason in decision['reasons'])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=output_dir.name + '.tmp.') as temporary:
        staging = Path(temporary)
        for split, value in [('train', kept), ('val', []), ('test', [])]:
            (staging / f'reasoning_chains_{split}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        (staging / 'review_curation_decisions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in decisions))
        # Rehash all source exports/audit bytes before publishing completion.
        for path, checksum in source.items():
            if file_sha256(path) != checksum:
                raise ValueError('Source artifact changed during curation')
        result = {'schema_version': 1, 'curation_version': VERSION, 'source_tool_sha256': file_sha256(__file__), 'source_quality_manifest_id': parent['manifest_id'],
                  'source_quality_policy_id': parent.get('verifier_policy_id'), 'audit_id': audit_id,
                  'audit_path': str(audit_path), 'audit_sha256': source[str(audit_path)],
                  'input_chains': len(entries), 'input_questions': len(original_questions),
                  'reviewed_exact_chains': len(reviewed), 'reviewed_questions': len(audits_by_question),
                  'flagged_questions': len(flagged), 'excluded_chains': len(entries) - len(kept),
                  'retained_chains': len(kept), 'retained_questions': len(retained_questions),
                  'flagged_question_reasons': flagged, 'reason_question_counts': dict(reason_questions),
                  'reason_chain_counts': dict(reason_chains), 'source_artifact_sha256': source,
                  'outputs': {path.name: file_sha256(path) for path in staging.iterdir()},
                  'policy': 'Quarantine every accepted variant of a question when any exact reviewed chain has non-strong support, non-supported answer, non-supported localization, source-label ambiguity, repeated evidence or explicit unsupported/unverifiable claims.',
                  'critic_policy_changed': False, 'official_labels_changed': False, 'source_exports_changed': False,
                  'visual_correctness_certified': False,
                  'limitations': ['Post-filter dataset curation is distinct from automated critic acceptance and locked validation.',
                                  'Only a sampled chain was inspected for each reviewed question; all accepted variants of flagged questions are conservatively withheld.',
                                  'Unreviewed retained chains are not certified correct. The original audit counts are unchanged; no post-curation error rate is estimated.']}
        result['manifest_id'] = content_hash(result)
        (staging / 'review_curation_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        if output_dir.exists():
            raise ValueError('Output directory appeared during curation')
        staging.rename(output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = apply_review(args.input_dir, args.audit, args.output_dir)
    print(json.dumps({key: result[key] for key in ['input_chains', 'input_questions', 'flagged_questions',
                                                 'excluded_chains', 'retained_chains', 'retained_questions', 'manifest_id']}, indent=2))


if __name__ == '__main__':
    main()
