import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from app.ml.baseline_data import MODEL_FILES, load_classification_split, load_splits, prepare_ticket_text
from app.ml.baseline_models import run_baseline, train_category_classifier, train_priority_classifier
from app.ml.dataset_audit import CATEGORIES, PRIORITIES
from app.ml.evaluation import METRICS, evaluate_classifier
from app.ml.preprocessing import COLUMNS
from app.services.classification_service import ClassificationService


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    for split in ('train', 'validation', 'test'):
        with (directory / f'{split}.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for category in CATEGORIES:
                for priority in PRIORITIES:
                    for i in range(3):
                        writer.writerow(dict(ticket_id=f'{split}-{category}-{priority}-{i}',
                                             ticket_text=f'{category} support request {priority} urgency {split}onlytoken example {i}.',
                                             category=category, priority=priority))
    # Must never become a feature source, regardless of content.
    (directory / 'historical_tickets.csv').write_text('answer\nsecretanswerword\n', encoding='utf-8')
    return directory


@pytest.fixture(scope='module')
def saved_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp('real-models')
    texts, categories, priorities = [], [], []
    for category in CATEGORIES:
        for priority in PRIORITIES:
            for i in range(3):
                texts.append(f'{category} support issue, {priority} urgency, reference {i}!')
                categories.append(category)
                priorities.append(priority)
    models = {'category': train_category_classifier(texts, categories),
              'priority': train_priority_classifier(texts, priorities)}
    for target, model in models.items():
        joblib.dump(model, directory / MODEL_FILES[target])
    return directory


def test_processed_split_loading_and_all_labels(data_dir):
    splits, checked_audit = load_splits(data_dir)
    assert not checked_audit
    assert set(splits) == {'train', 'validation', 'test'}
    for split in splits.values():
        assert len(split.texts) == 90
        assert set(split.categories) == set(CATEGORIES)
        assert set(split.priorities) == set(PRIORITIES)


@pytest.mark.parametrize('column', COLUMNS)
def test_missing_required_column_fails(data_dir, column):
    path = data_dir / 'train.csv'
    with path.open('w', encoding='utf-8') as file:
        file.write(','.join(name for name in COLUMNS if name != column) + '\n')
    with pytest.raises(ValueError, match='expected exactly'):
        load_classification_split(path)


def test_answer_column_is_rejected(data_dir):
    path = data_dir / 'train.csv'
    content = path.read_text(encoding='utf-8')
    path.write_text(content.replace('ticket_id,ticket_text,category,priority',
                                    'ticket_id,ticket_text,category,priority,answer', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='answers/extra features are forbidden'):
        load_splits(data_dir)


@pytest.mark.parametrize('problem', ['duplicate_text', 'duplicate_id', 'unknown_category', 'unknown_priority', 'blank', 'missing_class'])
def test_invalid_or_leaking_splits_fail(data_dir, problem):
    path = data_dir / 'validation.csv'
    with path.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    with (data_dir / 'train.csv').open(encoding='utf-8', newline='') as file:
        first_train = next(csv.DictReader(file))
    if problem == 'duplicate_text':
        rows[0]['ticket_text'] = '  ' + first_train['ticket_text'].replace(' ', '\t') + '  '
    elif problem == 'duplicate_id':
        rows[0]['ticket_id'] = first_train['ticket_id']
    elif problem == 'unknown_category':
        rows[0]['category'] = 'Invented'
    elif problem == 'unknown_priority':
        rows[0]['priority'] = 'Critical'
    elif problem == 'blank':
        rows[0]['ticket_text'] = ' \n\t '
    else:
        rows = [row for row in rows if row['category'] != CATEGORIES[0]]
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_splits(data_dir)


def test_missing_split_and_mismatched_audit_fail(data_dir):
    with pytest.raises(FileNotFoundError, match='Run preprocessing first'):
        load_classification_split(data_dir / 'absent.csv')
    (data_dir / 'dataset_audit.json').write_text('{"output_sha256": {}}', encoding='utf-8')
    with pytest.raises(ValueError, match='differs from the audited split'):
        load_splits(data_dir)


def test_training_only_fit_and_serialized_artifacts(data_dir, tmp_path, monkeypatch):
    splits, _ = load_splits(data_dir)
    fitted = []
    original_fit = Pipeline.fit

    def recording_fit(self, texts, labels, *args, **kwargs):
        fitted.append((list(texts), list(labels)))
        return original_fit(self, texts, labels, *args, **kwargs)

    monkeypatch.setattr(Pipeline, 'fit', recording_fit)
    output = tmp_path / 'baseline'
    results = run_baseline(data_dir, output)
    train = splits['train']
    assert fitted == [(train.texts, train.categories), (train.texts, train.priorities)]
    for target in ('category', 'priority'):
        model = joblib.load(output / 'models' / MODEL_FILES[target])
        vocabulary = model.named_steps['tfidf'].vocabulary_
        assert all(token not in vocabulary for token in ('validationonlytoken', 'testonlytoken', 'secretanswerword'))
        assert model.named_steps['classifier'].class_weight is None
        assert model.named_steps['tfidf'].max_features == 50000
        assert (output / f'{target}_confusion_matrix.png').read_bytes().startswith(b'\x89PNG')
        assert (output / f'{target}_confusion_matrix.csv').is_file()
        for name in splits:
            measured = results[target][name]
            assert all(0 <= measured[key] <= 1 for key in METRICS)
            assert sum(sum(row) for row in measured['confusion_matrix']) == 90
            assert set(model.classes_) <= set(measured['classification_report'])
    assert json.loads((output / 'baseline_results.json').read_text())['leakage_checks']['answer_loaded'] is False
    assert (output / 'baseline_results.md').is_file()
    assert ClassificationService(output / 'models').predict('Please investigate a billing issue.').category in CATEGORIES


def test_evaluation_does_not_refit_or_mutate(saved_models):
    model = joblib.load(saved_models / MODEL_FILES['category'])
    idf = model.named_steps['tfidf'].idf_.copy()
    coef = model.named_steps['classifier'].coef_.copy()
    vocabulary = model.named_steps['tfidf'].vocabulary_.copy()
    result = evaluate_classifier(model, ['heldoutuniquetoken'] * 10, list(CATEGORIES))
    np.testing.assert_array_equal(idf, model.named_steps['tfidf'].idf_)
    np.testing.assert_array_equal(coef, model.named_steps['classifier'].coef_)
    assert vocabulary == model.named_steps['tfidf'].vocabulary_
    assert len(result['labels']) == 10


@pytest.mark.parametrize('text', [None, 12, '', ' \n\t '])
def test_inference_rejects_invalid_text(saved_models, text):
    with pytest.raises(ValueError, match='nonempty string'):
        ClassificationService(saved_models).predict(text)


@pytest.mark.parametrize('text', ['Please help with billing.', 'Error #404: cannot log in! Why?', 'Unseen vocabulary zzyxq.'])
def test_real_inference_probabilities_determinism_and_labels(saved_models, text):
    service = ClassificationService(saved_models)
    first = service.predict(text)
    assert first == service.predict(text)
    assert first.category in CATEGORIES and first.priority in PRIORITIES
    for target in ('category', 'priority'):
        model = service.models[target]
        probabilities = model.predict_proba([prepare_ticket_text(text)])[0]
        index = list(model.classes_).index(getattr(first, target))
        assert getattr(first, f'{target}_confidence') == pytest.approx(probabilities[index])
        assert 0 <= getattr(first, f'{target}_confidence') <= 1


def test_missing_and_invalid_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing category model'):
        ClassificationService(tmp_path)
    joblib.dump({'not': 'a pipeline'}, tmp_path / MODEL_FILES['category'])
    with pytest.raises(ValueError, match='Cannot load valid category model'):
        ClassificationService(tmp_path)


def test_training_reproducibility(saved_models):
    texts = [f'{label} priority request number {i}' for label in PRIORITIES for i in range(5)]
    labels = [label for label in PRIORITIES for _ in range(5)]
    first = train_priority_classifier(texts, labels)
    second = train_priority_classifier(texts, labels)
    np.testing.assert_array_equal(first.predict_proba(texts), second.predict_proba(texts))
