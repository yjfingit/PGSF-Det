import os
import subprocess
import unittest
from unittest.mock import patch
from pgsf.utils.gpu_select import select_gpus


class GPUSelection(unittest.TestCase):
    def run_query(self,*args,**kwargs):
        cmd=args[0]
        output=('0, gpu-a, RTX 3080, 10240, 1\n1, gpu-b, RTX 3080, 10240, 8050\n'
                '2, gpu-c, RTX 3080, 10240, 1\n3, gpu-d, RTX 3080, 10240, 200\n'
                if 'index,uuid' in cmd[1] else 'gpu-b, 123\ngpu-d, 234\n')
        return subprocess.CompletedProcess(cmd,0,stdout=output)

    @patch('pgsf.utils.gpu_select.subprocess.run')
    def test_auto_excludes_busy_process_even_with_low_memory(self,run):
        run.side_effect=self.run_query
        with patch.dict(os.environ,{},clear=True):selected,_=select_gpus()
        self.assertEqual([g['index'] for g in selected],[0,2])

    @patch('pgsf.utils.gpu_select.subprocess.run')
    def test_visible_subset_and_no_gpu_failure(self,run):
        run.side_effect=self.run_query
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2'},clear=True):
            selected,_=select_gpus();self.assertEqual([g['index'] for g in selected],[2])
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'},clear=True):
            with self.assertRaises(RuntimeError):select_gpus()

    @patch('pgsf.utils.gpu_select.subprocess.run')
    def test_explicit_gpu_validation(self,run):
        run.side_effect=self.run_query
        with patch.dict(os.environ,{},clear=True):
            selected,_=select_gpus('2,0');self.assertEqual([g['index'] for g in selected],[2,0])
            with self.assertRaises(RuntimeError):select_gpus('1')
            with self.assertRaises(ValueError):select_gpus('0,0')
            with self.assertRaises(ValueError):select_gpus('9')


if __name__=='__main__':unittest.main()
