#!/usr/bin/env python
"""Read-only, clean epoch tables from running DDP console logs.

python scripts/watch_training.py            # live, latest 3 epochs
python scripts/watch_training.py --once     # one snapshot
python scripts/watch_training.py --last 10  # latest 10 across both stages

Manual launches (calling scripts/train.py directly, so there is no run directory
with status.json and stageN_console.log) are supported too::

python scripts/watch_training.py --log logs/my_launch.log --stage 2

Requires only Python's standard library; it never imports torch or uses a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
TRAIN = re.compile(r'^\s*\d+/\d+\s+\d+(?:\.\d+)?G\s+')
PROGRESS = re.compile(
    r'^(.*?):\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*'
    r'\[([^<]+)<[^,]+,\s*([\d.]+)(it/s|s/it)\]'
)


def pretty_progress(line: str) -> str:
    """Keep real losses/counts; render a compact bar and seconds per iteration."""
    match = PROGRESS.match(line)
    if not match:
        return line
    desc, percent, done, total, elapsed, rate, unit = match.groups()
    percent = int(percent)
    filled = min(12, round(percent * 12 / 100))
    bar = '━' * filled + '─' * (12 - filled)
    value = float(rate)
    seconds_per_iter = 1 / value if unit == 'it/s' and value > 0 else value
    parts = [int(v) for v in elapsed.split(':')]
    seconds = sum(value * 60**i for i, value in enumerate(reversed(parts)))
    return (f'{desc}: {percent:3d}% {bar} {done}/{total} '
            f'{seconds_per_iter:.2f}s/it {seconds:.1f}s')


def aligned_table(headers: list[str], rows: list[list[str]], width: int) -> list[str]:
    """Split columns into groups at narrow widths, retaining every value."""
    sizes=[max([len(h)]+[len(row[i]) for row in rows]) for i,h in enumerate(headers)]
    groups=[];group=[];used=0
    for i,size in enumerate(sizes):
        if group and used+2+size>width:
            groups.append(group);group=[];used=0
        group.append(i);used+=size+(2 if len(group)>1 else 0)
    if group:groups.append(group)
    lines=[]
    for group in groups:
        if lines:lines.append('')
        lines.append('  '.join(headers[i].rjust(sizes[i]) for i in group))
        for row in rows:lines.append('  '.join(row[i].rjust(sizes[i]) for i in group))
    return lines


def responsive_block(lines: list[str], width: int) -> list[str]:
    """Render trainer records rather than reusing its fixed-width spacing."""
    header=next((line.split() for line in lines if 'GPU_mem' in line),None)
    train=next((line for line in lines if TRAIN.match(line)),None)
    rendered=[lines[0]]
    if header and train:
        values=train.split(':',1)[0].split()
        if len(values)==len(header):rendered.extend(aligned_table(header,[values],width))
        else:rendered.extend(textwrap.wrap(train,width=width))
        progress=pretty_progress(train)
        suffix=progress.split(':',1)[-1].strip()
        rendered.append('Train '+suffix)
    elif header:
        rendered.extend(aligned_table(header,[],width))
        rendered.append('Waiting for first batch progress…')
    val=next((line for line in lines if line.lstrip().startswith('Class')),None)
    if val:rendered.append('Val   '+pretty_progress(val).split(':',1)[-1].strip())
    rows=[line.split() for line in lines if re.match(r'^\s*(RGB|TIR|mean)\s+\d+\s+\d+\s+',line)]
    if rows:rendered.extend(aligned_table(['Class','Images','Instances','Box(P','R','mAP50','mAP50-95)'],rows,width))
    for line in lines:
        if line.startswith('Epoch time '):rendered.extend(textwrap.wrap(line,width=width))
    # The progress bar has a fixed small budget; wrapping also supports tiny terminals.
    return [piece for line in rendered for piece in (textwrap.wrap(line,width=width,replace_whitespace=False,
             drop_whitespace=False) if len(line)>width else [line])]


def epoch_blocks(text: str, stage: int, width: int | None = None) -> list[str]:
    """Collapse carriage-return refreshes, hide framework initialization messages."""
    text = ANSI.sub('', text)
    chunks = re.split(r'(?m)^Stage \d+ \|[^\r\n]*[\r\n]+', text)[1:]
    result = []
    for chunk in chunks:
        lines = [line.rstrip() for line in re.split(r'[\r\n]+', chunk) if line.strip()]
        headers = [line for line in lines if line.lstrip().startswith('Epoch') and 'GPU_mem' in line]
        train = [line for line in lines if TRAIN.match(line)]
        val = [line for line in lines if line.lstrip().startswith('Class') and 'mAP50' in line]
        metrics = [line for line in lines if re.match(r'^\s*(RGB|TIR|mean)\s+\d+\s+\d+\s+', line)]
        summary = [line for line in lines if line.startswith('Epoch time ')]
        if not headers:
            continue
        displayed = [f'Stage {stage}', headers[-1]]
        displayed.append(pretty_progress(train[-1]) if train else 'Waiting for first batch progress…')
        if val:
            displayed.append(pretty_progress(val[-1]))
        # A resumed epoch can occur twice: keep the most recent metric of each name.
        last_metrics = {line.split()[0]: line for line in metrics}
        for name in ('RGB', 'TIR', 'mean'):
            if name in last_metrics:
                displayed.append(last_metrics[name])
        if summary:
            displayed.append(summary[-1])
        result.append('\n'.join(responsive_block(displayed,width) if width else displayed))
    return result


def snapshot(run_dir: Path, last: int, width: int | None = None,
             log: Path | None = None, stage: int | None = None) -> tuple[str, bool]:
    status_path = run_dir / 'status.json' if run_dir else None
    try:
        status = json.loads(status_path.read_text()) if status_path else {}
    except (OSError, ValueError):
        status = {}
    state = status.get('state', 'waiting')
    pid = status.get('pid')
    if state == 'running' and pid:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            state = 'process stopped; inspect console log'
        except PermissionError:
            pass
    blocks = []
    # Manual launches (scripts/train.py directly) have no run-dir layout, so the
    # caller may point at one console log and name its stage explicitly.
    if log is not None:
        if log.exists():
            blocks.extend(epoch_blocks(log.read_text(errors='replace'),
                                       stage or 2, width))
    else:
        for stage_id in (1, 2):
            path = run_dir / f'stage{stage_id}_console.log'
            if path.exists():
                blocks.extend(epoch_blocks(path.read_text(errors='replace'), stage_id, width))
    budget = status.get('epoch_budget', '?')
    patience = status.get('early_stop_patience', '?')
    title = (f'PGSF-Det | GPUs {status.get("physical_gpus", [0, 7])} | '
             f'{state} | stage {status.get("stage", "?")} | '
             f'budget {budget} epochs | patience {patience}')
    body = '\n\n'.join(blocks[-last:]) if blocks else 'Waiting for training logs…'
    if state == 'failed':
        path = run_dir / f'stage{status.get("stage", 1)}_console.log'
        if path.exists():
            body += '\n\nError log tail:\n' + '\n'.join(path.read_text(errors='replace').splitlines()[-15:])
    if width:title='\n'.join(textwrap.wrap(title,width))
    return title + '\n\n' + body, state in ('complete', 'failed','stopped') or state.startswith('process stopped')


def terminal_frame(lines: list[str]) -> str:
    """Address and erase every row: shorter replacements leave no stale suffix.

    Absolute cursor positions avoid newline/autowrap behavior across terminals.
    Clear the remaining screen when the new frame has fewer rows than the old.
    """
    frame=''.join(f'\033[{row};1H\033[2K{line}' for row,line in enumerate(lines,1))
    return frame+f'\033[{len(lines)+1};1H\033[J'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path,help='default: latest launched training run')
    parser.add_argument('--once', action='store_true', help='print once and exit')
    parser.add_argument('--last', type=int, default=3, help='show latest N epoch blocks')
    parser.add_argument('--interval', type=float, default=.2, help='refresh seconds (default 0.2)')
    parser.add_argument('--width',type=int,help='override detected terminal columns')
    parser.add_argument('--log', type=Path,
                        help='watch this console log directly (manual launches that '
                             'have no run-dir layout); disables status.json reading')
    parser.add_argument('--stage', type=int, choices=(1, 2),
                        help='stage label for --log (default 2)')
    args = parser.parse_args()
    if args.log is not None:
        if not args.log.is_file():
            parser.error(f'log file does not exist: {args.log}')
        # No run directory is required in this mode; status.json stays absent.
        args.run_dir = None
    elif args.run_dir is None:
        latest=ROOT/'outputs/latest_training_run.json'
        args.run_dir=(Path(json.loads(latest.read_text())['run_dir']) if latest.exists()
                      else ROOT/'outputs/fulltrain_20261010_200e')
    if args.last < 1 or args.interval <= 0:
        parser.error('--last and --interval must be positive')
    if args.width is not None and args.width<20:
        parser.error('--width must be at least 20')
    if args.run_dir is not None and not args.run_dir.is_dir():
        parser.error(f'run directory does not exist: {args.run_dir}')
    previous = None
    interactive=sys.stdout.isatty() and not args.once
    if interactive:
        # Alternate screen isolates the live viewer from terminal scrollback.
        sys.stdout.write('\033[?1049h\033[?25l\033[2J')
        sys.stdout.flush()
    try:
        while True:
            columns=args.width or max(20,shutil.get_terminal_size((120,40)).columns-1)
            output, finished = snapshot(args.run_dir, args.last, columns,
                                        args.log, args.stage)
            if interactive:
                terminal_height=shutil.get_terminal_size((120,40)).lines
                screen=output.splitlines()
                if len(screen)>terminal_height-1:
                    # Prefer the active epoch at small heights; avoid scrolling on redraw.
                    output,_=snapshot(args.run_dir,1,columns,args.log,args.stage)
                    screen=output.splitlines()
                screen=[line[:columns] for line in screen[:max(1,terminal_height-1)]]
                sys.stdout.write(terminal_frame(screen))
                sys.stdout.flush()
            elif output != previous:
                print(output, flush=True)
                previous = output
            if args.once or finished:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        # Stopping the viewer never interrupts training.
        pass
    finally:
        if interactive:
            sys.stdout.write('\033[?25h\033[?1049l')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
