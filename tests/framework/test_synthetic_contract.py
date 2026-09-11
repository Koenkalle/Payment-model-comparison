"""Every scenario must be valid input to both native and browser comparisons."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.implementations.payment_json import normalize


class ScenarioContractTests(unittest.TestCase):
    def test_all_scenarios_sizes_and_seeds_are_payment_documents(self):
        script = "const s=require('./datasets/implementations/synthetic_payments');process.stdout.write(JSON.stringify(s.catalog.flatMap(x=>['small','medium','large'].flatMap(size=>[42,314].map(seed=>s.build(x.id,size,seed))))));"
        documents = json.loads(subprocess.check_output(['node', '-e', script], cwd=ROOT, text=True))
        self.assertEqual(len(documents), 30)
        for document in documents:
            with self.subTest(scenario=document['name'], size=document['size'], seed=document['seed']):
                normalized = normalize(document)
                self.assertEqual(len(normalized['events']), len(document['events']))
                payments = {event['id'] for event in document['events'] if event['kind'] == 'payment'}
                self.assertEqual(set(normalized['truth']), payments)
                self.assertEqual(normalized['truth'], document['truth'])

    def test_checked_in_scenario_files_import_without_repairs(self):
        paths = sorted(ROOT.joinpath('datasets').glob('*-*-*.json'))
        self.assertTrue(paths)
        self.assertEqual({path.name.split('-')[0] for path in paths}, {'relay', 'split', 'benign', 'takeover', 'mixed'})
        for path in paths:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text())
                self.assertEqual(normalize(document)['truth'], document['truth'])


if __name__ == '__main__':
    unittest.main()
