"""Select idle physical GPUs before CUDA initializes; no torch dependency."""
import csv
import io
import os
import subprocess


def select_gpus(request='auto',max_used_mib=512,min_free_mib=9000):
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used',
                         '--format=csv,noheader,nounits'],check=True,text=True,capture_output=True).stdout
    inventory=[]
    for row in csv.reader(io.StringIO(rows)):
        index,uuid,name,total,used=[value.strip() for value in row]
        inventory.append({'index':int(index),'uuid':uuid,'name':name,'total_mib':int(total),
                          'used_mib':int(used),'free_mib':int(total)-int(used)})
    process_rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'],check=True,text=True,capture_output=True).stdout
    active={row[0].strip() for row in csv.reader(io.StringIO(process_rows)) if len(row)>=2}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed=None if visible is None else {v.strip() for v in visible.split(',') if v.strip()}
    def permitted(g):return allowed is None or str(g['index']) in allowed or g['uuid'] in allowed
    if request=='auto':
        selected=[g for g in inventory if permitted(g) and g['used_mib']<=max_used_mib
                  and g['free_mib']>=min_free_mib and g['uuid'] not in active]
    else:
        tokens=request.split(',')
        if not tokens or any(not v.strip().isdigit() for v in tokens):
            raise ValueError('--gpus must be auto or comma-separated physical GPU indices')
        indices=[int(v) for v in tokens]
        if len(set(indices))!=len(indices):raise ValueError('Duplicate GPU indices')
        by_index={g['index']:g for g in inventory}
        if any(i not in by_index for i in indices):raise ValueError('Unknown physical GPU index')
        selected=[by_index[i] for i in indices]
        if any(not permitted(g) for g in selected):raise ValueError('Selected GPU is outside CUDA_VISIBLE_DEVICES')
        busy=[g['index'] for g in selected if g['uuid'] in active or g['used_mib']>max_used_mib or g['free_mib']<min_free_mib]
        if busy:raise RuntimeError(f'Selected GPU(s) occupied or insufficient memory: {busy}')
    if not selected:
        raise RuntimeError('No idle GPU has enough memory for batch8; training was not started')
    return selected,inventory
