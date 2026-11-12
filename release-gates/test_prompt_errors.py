import sys
import types
import unittest
from unittest.mock import Mock, patch

import tasks


class PromptErrorTest(unittest.TestCase):
    def test_empty_and_raising_prompts_preserve_positions_and_continue(self):
        dataset = Mock()
        dataset.shuffle.return_value.__getitem__ = Mock(return_value={
            'context': ['ok', 'empty', 'raises', 'last'],
            'question': ['q'] * 4,
        })
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kw: text
        tokenizer.convert_ids_to_tokens.side_effect = lambda value: value
        tokenizer.convert_tokens_to_ids.side_effect = lambda value: value
        tokenizer.decode.side_effect = lambda value, **kw: ''.join(value)
        seen = []
        def translate(batch, **kwargs):
            text = batch[0]
            seen.append(text)
            if 'raises' in text:
                raise RuntimeError('inference error')
            tokens = ['   '] if 'empty' in text else ['answer']
            return [types.SimpleNamespace(hypotheses=[tokens])]
        translator = types.SimpleNamespace(translate_batch=translate, device='cpu',
                                           compute_type='int8_float32')
        datasets = types.SimpleNamespace(load_dataset=Mock(return_value=dataset))
        transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=Mock(return_value=tokenizer)))
        ct2 = types.SimpleNamespace(Translator=Mock(return_value=translator))
        with patch.dict(sys.modules, datasets=datasets, transformers=transformers, ctranslate2=ct2):
            result = tasks.run_t5gemma_generation(limit=4, model_dir='model')
        self.assertEqual(result['outputs'], ['answer', '', '', 'answer'])
        self.assertEqual([e['prompt'] for e in result['prompt_errors']], [2, 3])
        self.assertIn('empty completion', result['prompt_errors'][0]['error'])
        self.assertIn('inference error', result['prompt_errors'][1]['error'])
        self.assertEqual(len(seen), 6)  # Two warm-up prompts plus all four measured prompts.
        self.assertGreater(result['tokens_per_sec'], 0)


if __name__ == '__main__':
    unittest.main()
