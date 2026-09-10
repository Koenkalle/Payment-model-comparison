"""Check registry-driven outputs and the offline delivery contract after building."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.build import validate_registry
from framework.registry import browser_scripts


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.scripts, self.links, self.current = [], [], None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.current = {'attrs': attrs, 'text': ''}
            self.scripts.append(self.current)
        elif tag == 'a' and attrs.get('aria-current') == 'page':
            self.links.append(attrs['href'])

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def handle_endtag(self, tag):
        if tag == 'script':
            self.current = None


class BuildTests(unittest.TestCase):
    def test_registered_outputs_are_self_contained(self):
        registry = json.loads((ROOT / 'tools/registry.json').read_text())
        catalog = json.loads((ROOT / 'designs.json').read_text())
        validate_registry(registry)
        for tool in registry['tools']:
            with self.subTest(tool=tool['id']):
                page = Page((ROOT / tool['output']).read_text())
                self.assertTrue(all('src' not in script['attrs'] for script in page.scripts))
                payloads = [script for script in page.scripts
                            if script['attrs'].get('id') == tool['model_data_id']]
                self.assertEqual(len(payloads), 1)
                payload = json.loads(payloads[0]['text'])
                wanted = ([design['id'] for design in catalog['designs']]
                          if tool['model_ids'] == 'all' else tool['model_ids'])
                self.assertEqual([model['id'] for model in payload['models']], wanted)
                self.assertEqual(page.links, [tool['output']])
                sources = [(ROOT / name).read_text() for name in
                           browser_scripts() + registry['shared_scripts'] + tool['scripts']]
                embedded = [script['text'].strip() for script in page.scripts]
                positions = [embedded.index(source.strip()) for source in sources]
                self.assertEqual(positions, sorted(positions))
                if tool.get('explanation'):
                    self.assertEqual(payload['explanation'], json.loads((ROOT / tool['explanation']).read_text()))

    def test_duplicate_tools_are_rejected(self):
        registry = json.loads((ROOT / 'tools/registry.json').read_text())
        registry['tools'].append(registry['tools'][0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_registry(registry)


if __name__ == '__main__':
    unittest.main()
