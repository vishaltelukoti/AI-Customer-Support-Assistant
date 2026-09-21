import csv
import json
from copy import deepcopy

import pytest

from app.ml.dataset_audit import (
    CATEGORIES, PRIORITIES, REQUIRED, audit_dataset, clean_text, combine_text,
    duplicate_audit, load_csv, mask_pii, normalize_text, select_english,
)
from app.ml.preprocessing import (
    COLUMNS, assign_groups, identity_keys, prepare_records, preprocess_dataset,
    split_by_category, verify_splits,
)


@pytest.fixture
def rows():
    # Alphabetic variations are distinct content, not number-only templates.
    return [dict(subject=f"{category} issue {chr(65 + i // 26)}{chr(65 + i % 26)}",
                 body=f"The {category} product {chr(65 + i // 26)}{chr(65 + i % 26)} cannot start. Error 404.",
                 answer="Agent response, excluded from input.", type="Incident", queue=category,
                 priority=PRIORITIES[i % 3], language="en", version=("51", "52", "400")[i % 3])
            for category in CATEGORIES[:4] for i in range(30)]


def write_csv(path, rows, columns=REQUIRED):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("value, expected", [
    (None, ""), (float("nan"), ""), (42, "42"), (" \n\t ", ""),
    ("  My\t payment\n failed  ", "My payment failed"),
    (" Don't remove #123, $42.50 or café! ", "Don't remove #123, $42.50 or café!"),
])
def test_text_normalization(value, expected):
    assert normalize_text(value) == expected


def test_paragraph_cleaning_combination_and_missing_text():
    assert clean_text("  First\r\n\r\nSecond\t line \n\n\nLast ") == "First\n\nSecond line\n\nLast"
    assert clean_text(r"Dear Team,\n\nHelp\tplease") == "Dear Team,\n\nHelp please"
    assert clean_text(r"Cannot open C:\new\test.txt") == r"Cannot open C:\new\test.txt"
    assert combine_text(" Subject ", "Body\n\nDetails") == "Subject\n\nBody\n\nDetails"
    assert combine_text(None, "Body") == "Body"
    assert combine_text("Subject", None) == "Subject"
    assert combine_text(None, None) == ""


def test_csv_loading_with_quoted_multiline_text(tmp_path, rows):
    rows[0]['body'] = 'Message, with comma\nsecond line'
    source = write_csv(tmp_path / 'raw.csv', rows)
    columns, loaded = load_csv(source)
    assert columns == list(REQUIRED)
    assert loaded == rows


@pytest.mark.parametrize("column", REQUIRED)
def test_required_field_detection(tmp_path, column):
    source = write_csv(tmp_path / 'raw.csv', [], [name for name in REQUIRED if name != column])
    with pytest.raises(ValueError, match=f"Missing required columns: {column}"):
        load_csv(source)


def test_bad_csv_schema_and_empty_file(tmp_path, rows):
    source = write_csv(tmp_path / 'raw.csv', [])
    with pytest.raises(ValueError, match='no data'):
        load_csv(source)
    source.write_text(','.join(REQUIRED + ('subject',)) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate CSV column'):
        load_csv(source)
    source = write_csv(source, rows)
    with source.open('a', encoding='utf-8') as file:
        file.write('malformed,row\n')
    with pytest.raises(ValueError, match='header width'):
        load_csv(source)


def test_language_filter_and_missing_language(rows):
    rows[0]['language'] = 'de'
    selected = select_english(rows)
    assert len(selected) == len(rows) - 1
    assert selected[0][0] == 2  # Original 1-based data record retained.
    assert all(row['language'] == 'en' for _, row in selected)
    for invalid in ('', 'English', 'EN', 'unknown'):
        rows[0]['language'] = invalid
        with pytest.raises(ValueError, match='language'):
            select_english(rows)


def test_version_overlap_exact_rows_and_source_ids(rows):
    rows = rows[:4]
    for i, row in enumerate(rows):
        row['ticket_id'] = str(i)
    rows.append(deepcopy(rows[0]))
    copy = deepcopy(rows[0])
    copy['version'] = '400'
    rows.append(copy)
    report = duplicate_audit(rows, list(REQUIRED) + ['ticket_id'])
    assert report['exact_rows']['excess_rows'] == 1
    assert report['subject_body']['excess_rows'] == 2
    assert report['normalized_ticket_text']['cross_version_groups'] == 1
    assert report['normalized_subject']['cross_version_rows'] == 3
    assert report['ticket_id']['excess_rows'] == 2
    assert report['ticket_id']['version_pair_shared_keys'] == {'400|51': 1}


def test_read_only_audit_schema_statistics_and_leakage(tmp_path, rows):
    rows[0]['subject'] = ''
    source = write_csv(tmp_path / 'raw.csv', rows)
    before = source.read_bytes()
    audit = audit_dataset(source, list(REQUIRED), rows)
    assert audit['dataset']['rows'] == len(rows)
    assert audit['field_mapping']['ticket_id'] is None
    assert audit['schema'][0]['missing_count'] == 1
    assert audit['english_text_quality']['subject']['empty_after_cleaning'] == 1
    assert {field['field'] for field in audit['leakage_fields'] if field['classification_input']} == {'subject', 'body'}
    assert source.read_bytes() == before


@pytest.mark.parametrize('field,value', [
    ('queue', ''), ('queue', 'Other'), ('queue', ' technical support '),
    ('priority', ''), ('priority', 'High'), ('priority', 'critical'), ('version', ''),
])
def test_invalid_labels_reported_without_relabelling(rows, field, value):
    rows[0][field] = value
    prepared, summary = prepare_records(rows, 'a' * 64)
    assert len(prepared) == len(rows) - 1
    assert summary['invalid_records'][0]['source_record'] == 1
    assert any(field in reason for reason in summary['invalid_records'][0]['reasons'])
    assert rows[0][field] == value


def test_all_actual_categories_and_priority_values_accepted():
    rows = [dict(subject=f'{category} {priority}', body='Valid body.', answer='', type='Request',
                 queue=category, priority=priority, language='en', version='400')
            for category in CATEGORIES for priority in PRIORITIES]
    prepared, summary = prepare_records(rows, 'b' * 64)
    assert len(prepared) == len(rows)
    assert not summary['invalid_records']
    assert {row['category'] for row in prepared} == set(CATEGORIES)
    assert {row['priority'] for row in prepared} == set(PRIORITIES)


def test_empty_text_removed_but_short_and_subject_only_retained(rows):
    rows[0].update(subject='', body=' \n\t ')
    rows[1].update(subject='...', body='!!!')
    rows[2].update(subject='', body='Help')
    rows[3].update(subject='Subject only', body='')
    prepared, summary = prepare_records(rows, 'a' * 64)
    assert len(summary['invalid_records']) == 2
    assert prepared[0]['ticket_text'] == 'Help'
    assert prepared[1]['ticket_text'] == 'Subject only'


def test_first_valid_deduplication_and_conflicting_labels(rows):
    invalid = {**rows[0], 'queue': 'unsupported'}
    duplicate = {**rows[0], 'body': '  ' + rows[0]['body'] + '\t', 'version': '400'}
    prepared, summary = prepare_records([invalid] + rows + [duplicate], 'a' * 64)
    assert len(prepared) == len(rows)
    assert prepared[0]['source_record'] == '2'
    assert len(summary['invalid_records']) == 1
    assert len(summary['duplicate_records_removed']) == 1
    duplicate['priority'] = 'high'
    with pytest.raises(ValueError, match='Conflicting labels'):
        prepare_records(rows + [duplicate], 'a' * 64)


@pytest.mark.parametrize('text, kind', [
    ('Email customer@example.com please', 'email'), ('Visit https://example.com/help', 'url'),
    ('Phone +1 212-555-0199', 'phone'), ('Call +442079460123', 'phone'),
    ('Card 4111 1111 1111 1111', 'card_like'), ('Account number: AB12345678', 'account_number'),
    ('Reference 1234567890', 'long_numeric_id'),
])
def test_pattern_masking(text, kind):
    masked, counts = mask_pii(text)
    assert counts[kind] == 1
    assert f'[{kind.upper()}]' in masked
    assert mask_pii(masked)[0] == masked


def test_masking_deduplicates_before_splitting_and_masks_answer(rows):
    rows[0]['body'] = 'Please contact alice@example.com about the failing product.'
    rows[0]['answer'] = 'Contact agent@example.com or call 212-555-0199.'
    duplicate = {**rows[0], 'body': 'Please contact bob@example.com about the failing product.'}
    prepared, summary = prepare_records(rows + [duplicate], 'a' * 64)
    assert len(prepared) == len(rows)
    assert len(summary['duplicate_records_removed']) == 1
    assert summary['pii']['ticket_text']['email']['occurrences'] == 2
    assert '[EMAIL]' in prepared[0]['answer'] and '[PHONE]' in prepared[0]['answer']
    assert 'agent@example.com' not in prepared[0]['ticket_text']


def test_grouped_stratification_determinism_and_no_leakage(rows):
    # Different subjects, versions and labels with a shared body must stay together.
    rows[1]['body'] = rows[0]['body']
    prepared, _ = prepare_records(rows, 'a' * 64)
    stats = assign_groups(prepared)
    assert stats['multi_record_groups'] == 1
    splits, notes = split_by_category(prepared)
    assert splits == split_by_category(prepared)[0]
    assert splits != split_by_category(prepared, seed=17)[0]
    assert notes and prepared[0] in splits['train'] and prepared[1] in splits['train']
    assert sum(map(len, splits.values())) == len(prepared)
    assert verify_splits(splits)['cross_split_grouping_key_overlap'] == 0
    for split in splits.values():
        assert {row['category'] for row in split} == set(CATEGORIES[:4])
        assert {row['priority'] for row in split} == set(PRIORITIES)
    broken = deepcopy(splits)
    broken['test'].append({**broken['train'][0], 'ticket_id': 'different-id'})
    with pytest.raises(ValueError, match='leakage'):
        verify_splits(broken)


def test_duplicate_alias_keeps_source_id_chain_together(rows):
    rows[0]['ticket_id'] = 'source-A'
    duplicate = {**rows[0], 'ticket_id': 'source-B'}
    rows[1]['ticket_id'] = 'source-B'
    prepared, _ = prepare_records(rows + [duplicate], 'a' * 64)
    assign_groups(prepared)
    assert prepared[0]['group_id'] == prepared[1]['group_id']


def test_obvious_numeric_template_grouping(rows):
    rows[0]['body'] = 'The named accounting service fails with transaction error 404 and cannot restart.'
    rows[1]['body'] = 'The named accounting service fails with transaction error 502 and cannot restart.'
    prepared, _ = prepare_records(rows, 'a' * 64)
    assert len(prepared) == len(rows)  # Group, never delete based on a template.
    assign_groups(prepared)
    assert prepared[0]['group_id'] == prepared[1]['group_id']


def test_rare_strata_are_retained_and_reported(rows):
    rows = rows[:30] + [rows[30]] + rows[60:]
    prepared, _ = prepare_records(rows, 'a' * 64)
    assign_groups(prepared)
    splits, notes = split_by_category(prepared)
    assert any('independent groups' in note for note in notes)
    assert sum(map(len, splits.values())) == len(prepared)
    assert any(row['category'] == CATEGORIES[1] for row in splits['train'])


def test_pipeline_outputs_and_byte_repeatability(tmp_path, rows):
    source = write_csv(tmp_path / 'selected.csv', rows)
    original = source.read_bytes()
    output = tmp_path / 'processed'
    result = preprocess_dataset(source, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(before) == {'train.csv', 'validation.csv', 'test.csv', 'historical_tickets.csv', 'dataset_audit.json'}
    assert [result.audit['split']['sets'][name]['rows'] for name in ('train', 'validation', 'test')] == [84, 18, 18]
    for name in ('train', 'validation', 'test'):
        columns, exported = read_output(output / f'{name}.csv')
        assert columns == list(COLUMNS)
        assert all('Agent response' not in row['ticket_text'] for row in exported)
    history_columns, history = read_output(output / 'historical_tickets.csv')
    assert 'answer' in history_columns and 'split' in history_columns
    assert len(history) == len(rows)
    preprocess_dataset(source, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    assert source.read_bytes() == original
    audit = json.loads(before['dataset_audit.json'])
    assert audit['counts']['final_english'] == len(rows)


def read_output(path):
    with path.open(encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def test_failures_preserve_raw_and_existing_outputs(tmp_path, rows):
    source = write_csv(tmp_path / 'train.csv', rows)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='overwrite the raw source'):
        preprocess_dataset(source, tmp_path)
    assert source.read_bytes() == original
    rows[0]['language'] = ''
    source = write_csv(tmp_path / 'raw.csv', rows)
    with pytest.raises(ValueError, match='language'):
        preprocess_dataset(source, tmp_path)
    assert (tmp_path / 'train.csv').read_bytes() == original


def test_missing_minimum_categories_and_empty_data(rows):
    with pytest.raises(ValueError, match='four meaningful'):
        prepare_records(rows[:30], 'a' * 64)
    for row in rows:
        row['subject'] = row['body'] = ''
    with pytest.raises(ValueError, match='No valid English'):
        prepare_records(rows, 'a' * 64)
