import unittest
from scripts.watch_training import epoch_blocks, pretty_progress,aligned_table,terminal_frame


class TrainingViewer(unittest.TestCase):
    def test_refreshes_collapse_to_latest_and_do_not_invent_loss_fields(self):
        text = '''framework warning
Stage 1 | training augmentation
 Epoch GPU_mem box_loss cls_loss nwd_loss Instances Size
\r1/40 7.2G 1.0 0.5 0.2 16 800: 50%|xxxx        | 5/10 [00:02<00:02, 2.00it/s]
\r1/40 7.2G 0.8 0.4 0.1 16 800: 100%|xxxxxxxxxxxx| 10/10 [00:05<00:00, 2.00it/s]
\rClass Images Instances Box(P R mAP50 mAP50-95): 100%|xxxxxxxxxxxx| 4/4 [00:02<00:00, 2.00it/s]
 RGB 678 678 0.2 0.3 0.4 0.1
 TIR 678 678 0.3 0.4 0.5 0.2
 mean 678 1356 0.25 0.35 0.45 0.15
Epoch time 7.0s | skipped 0
'''
        blocks = epoch_blocks(text, 1)
        self.assertEqual(len(blocks), 1)
        self.assertNotIn('50%', blocks[0])
        self.assertNotIn('framework warning', blocks[0])
        self.assertNotIn('dfl_loss', blocks[0])
        self.assertIn('0.50s/it 5.0s', blocks[0])
        self.assertIn('RGB 678 678', blocks[0])
        self.assertIn('mean 678 1356', blocks[0])

    def test_duration_supports_hours_and_partial_progress(self):
        line = '1/2 7G: 25%|xxx         | 1/4 [1:02:03<00:09, 3.00s/it]'
        self.assertIn('3.00s/it 3723.0s', pretty_progress(line))

    def test_headers_and_values_share_column_endpoints_at_different_widths(self):
        headers=['Epoch','GPU_mem','box_loss','cls_loss','nwd_loss','Instances','Size']
        row=['8/40','7.21G','0.3344','0.0360','0.2720','16','800x1280/512x640']
        for width in (40,80,120,180):
            lines=aligned_table(headers,[row],width)
            self.assertTrue(all(len(line)<=width for line in lines))
            pairs=[line for line in lines if line]
            for i in range(0,len(pairs),2):
                ends=lambda line:[m.end() for m in __import__('re').finditer(r'\S+',line)]
                self.assertEqual(ends(pairs[i]),ends(pairs[i+1]))
        self.assertTrue(aligned_table(headers,[],80))

    def test_redraw_erases_each_row_and_old_rows_without_live_footer(self):
        lines=['Stage 1','  RGB 678 678 0.1980']
        frame=terminal_frame(lines)
        self.assertEqual(frame,'\033[1;1H\033[2KStage 1'
                         '\033[2;1H\033[2K  RGB 678 678 0.1980\033[3;1H\033[J')
        self.assertNotIn('Live ',frame)
        self.assertNotIn('viewer only',frame)
        self.assertNotIn('\n',frame)


if __name__ == '__main__':
    unittest.main()
