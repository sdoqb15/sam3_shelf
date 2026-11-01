import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sam3_shelf.training.preflight import preflight
from sam3_shelf.training.trainer import _train

class ScoreBudgetTests(unittest.TestCase):
 def test_fixed_budget_is_explicit_and_default_still_requires_full_cycle(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);model=root/'model';model.mkdir();mask=root/'mask';mask.write_bytes(b'mask')
   for name in ['config.json','model.safetensors','processor_config.json','tokenizer_config.json']:(model/name).write_bytes(b'fixture')
   def row(i):return dict(id=str(i),group=str(i),sampling_group='single',image_fingerprint=str(i),instances=[dict(mask=str(mask))])
   c=dict(method='supervised_joint',model=str(model),output_dir=str(root/'out'),data=dict(train='train',validation='val'),sampling=dict(multi_weight=8,single_weight=1),training=dict(minimum_updates=1,max_updates=1,validation_interval=1,gradient_accumulation=4))
   def manifest(path,root=None):return [row(i) for i in range(20)] if path=='train' else [row(21)]
   with patch('sam3_shelf.training.preflight.load_recipe',side_effect=lambda _:copy.deepcopy(c)),patch('sam3_shelf.training.preflight.read_manifest',side_effect=manifest),patch('sam3_shelf.training.preflight.validate_manifest',return_value={}),patch('sam3_shelf.training.preflight.assert_disjoint'):
    with self.assertRaises(ValueError):preflight(root/'recipe.json')
    c['training']['require_full_cycle']=False
    config,_,_,audit=preflight(root/'recipe.json');self.assertEqual(config['training']['effective_minimum_updates'],1);self.assertFalse(audit['require_full_cycle'])
 def test_prepared_preflight_cannot_be_used_with_another_recipe(self):
  c=dict(training=dict(effective_minimum_updates=1),model='a')
  with patch('sam3_shelf.config.load_recipe',return_value=dict(training={},model='different')):
   with self.assertRaisesRegex(ValueError,'different recipe'):_train('recipe.json',preflight_result=(c,[],[],{}))

if __name__=='__main__':unittest.main()
