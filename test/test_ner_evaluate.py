
from evaluation.evaluate import evaluate_ner_extraction, evaluate_ner_bio
import unittest


class TestNEREvaluation(unittest.TestCase):

    def test_evaluate_ner_extraction_perfect_match(self):
        preds = [[('Major Depressive Disorder', 'application-area')]]
        labels = [[('Major Depressive Disorder', 'application-area')]]
        results = evaluate_ner_extraction(preds, labels)
        self.assertEqual(results['accuracy_overall'], 1.0)
        self.assertEqual(results['f1_overall'], 1.0)

    def test_evaluate_ner_extraction_partial_match(self):
        preds  = [[('Major Depressive Disorder', 'application-area')]]
        labels = [[('Major Depressive Disorder', 'application-area'), ('MDD', 'application-area')]]
        results = evaluate_ner_extraction(preds, labels)
        self.assertEqual(results['accuracy_overall'], 0)
        self.assertEqual(results['precision_overall'], 1.0)
        self.assertEqual(results['recall_overall'], 0.5)
        self.assertAlmostEqual(results['f1_overall'], 0.6666, places=1)

    def test_evaluate_ner_extraction_no_match(self):
        preds = [[('Major Depressive Disorder', 'application-area')]]
        labels = [[('PTSD', 'application-area')]]
        results = evaluate_ner_extraction(preds, labels)
        self.assertEqual(results['accuracy_overall'], 0)

    def test_evaluate_ner_extraction_types(self):
        pred = [[('Major Depressive Disorder', 'application-area'), ('PTSD', 'application-area'), ('3 weeks', 'dosage'), ('5 mg/kg', 'dosage')]]
        labels = [[('Major Depressive Disorder', 'application-area'), ('PTSD', 'application-area'), ('5 mg/kg', 'dosage')]]
        results = evaluate_ner_extraction(pred, labels)
        self.assertEqual(results['precision_application-area'], 1.0)
        self.assertEqual(results['recall_application-area'], 1.0)
        self.assertEqual(results['f1_application-area'], 1.0)

        self.assertEqual(results['precision_dosage'], 0.5)
        self.assertEqual(results['recall_dosage'], 1.0)
        self.assertAlmostEqual(results['f1_dosage'], 0.6667, places=2)

    def test_evaluate_ner_extraction_duplicate_predictions(self):
        pred = [[('Major Depressive Disorder', 'application-area'), ('Major Depressive Disorder', 'application-area')]]
        true = [[('Major Depressive Disorder', 'application-area')]]
        results = evaluate_ner_extraction(pred, true)
        self.assertEqual(results['precision_application-area'], 1.0)
        self.assertEqual(results['recall_application-area'], 1.0)
        self.assertAlmostEqual(results['f1_application-area'], 1.0, places=2)


    def test_evaluate_ner_bio_basic(self):
        pred = [['B-Application area', 'B-Dosage', 'O']]
        true = [['B-Application area', 'B-Dosage', 'O']]
        results = evaluate_ner_bio(pred, true)
        self.assertEqual(results['f1 overall - strict'], 1.0)
        self.assertEqual(results['f1 APP - strict'], 1.0)
        self.assertEqual(results['f1 DOS - strict'], 1.0)