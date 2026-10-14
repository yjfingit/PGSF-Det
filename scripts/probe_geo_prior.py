#!/usr/bin/env python
"""Diagnostic: is the fitted affine prior actually doing anything at eval time?

Runs the trained geo-prior checkpoint three ways on the same split and prints
the metric for each. Nothing is retrained; only the affine buffers change.

  full    : the fitted affine (what training used)
  broken  : the affine deliberately corrupted (rotation + translation wrecked)
  identity: affine = identity, i.e. the model must pair raw normalised points

If the score barely moves between them, the prior is decorative.
"""
import argparse, json, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pgsf.config import load_config
from pgsf.engine.trainer import PGSFTrainer
from pgsf.models.detector import invert_affine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--out', default=None)
    a = p.parse_args()

    cfg = load_config(a.config, [])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = PGSFTrainer(cfg, device=device)
    tr.load_weights(a.ckpt)
    model = tr.ema.module if tr.ema else tr.model

    fitted = model._affine_A.detach().clone()
    variants = {
        'full_fitted': fitted,
        'identity': torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
        'broken': torch.tensor([[0.80, 0.45, 0.22],
                                [-0.38, 0.72, -0.15]]),
    }

    results = {}
    for name, A in variants.items():
        b = A[:, 2].clone()
        model.set_affine(A, b)
        if tr.ema:
            tr.ema.module.set_affine(A, b)
        m = tr.evaluate(split=a.split, use_ema=bool(tr.ema))
        key = 'mean' if 'mean' in m else None
        results[name] = {
            'rgb_AP50:95': m['rgb']['AP50:95'],
            'tir_AP50:95': m['tir']['AP50:95'],
            'mean_AP50:95': (m['rgb']['AP50:95'] + m['tir']['AP50:95']) / 2,
            'rgb_AP50': m['rgb']['AP50'], 'tir_AP50': m['tir']['AP50'],
            'match_acc': m.get('match', {}).get('match_acc'),
        }
        print(f"[{name:10s}] mean AP50:95 = {results[name]['mean_AP50:95']:.4f} "
              f"(RGB {results[name]['rgb_AP50:95']:.4f} / TIR {results[name]['tir_AP50:95']:.4f})"
              + (f"  match_acc {results[name]['match_acc']:.4f}"
                 if results[name]['match_acc'] is not None else ''),
              flush=True)

    base = results['full_fitted']['mean_AP50:95']
    print("\n=== delta vs fitted ===")
    for name in ('identity', 'broken'):
        print(f"{name:10s} {results[name]['mean_AP50:95'] - base:+.4f}")
    print(f"\nA_fitted =\n{fitted.numpy().round(4).tolist()}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {'ckpt': a.ckpt, 'split': a.split, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
