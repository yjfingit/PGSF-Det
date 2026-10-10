"""Fast UAV-CB statistics probe (sampled, no full-image decode)."""
import glob, json, os, collections, random
import numpy as np

B = "/home/yangjuanfeng/lab/shared/datasets/UAV-CB"
random.seed(0)
tot = 0
lab = collections.Counter()
nobj = collections.Counter()
wh = []
seq = collections.Counter()
missing = 0
center_delta = []
per_scene = {}
for scene in sorted(os.listdir(B)):
    vs = sorted(glob.glob(os.path.join(B, scene, "visible", "*.jpg")))
    sub = random.sample(vs, min(300, len(vs)))
    per_scene[scene] = len(vs)
    for v in sub:
        base = os.path.basename(v)
        t = os.path.join(B, scene, "thermal", base.replace("_visible_", "_thermal_"))
        if not os.path.exists(t):
            missing += 1
            continue
        tot += 1
        # sequence id = the "1" in xxx_visible_1_000001
        stem = os.path.basename(v)[:-4]
        parts = stem.split("_")
        seq[(scene, parts[-2])] += 1
        with open(v.replace(".jpg", ".json")) as f:
            d = json.load(f)
        n = len(d["shapes"])
        nobj[n] += 1
        for s in d["shapes"]:
            lab[s["label"]] += 1
            xs = [p[0] for p in s["points"]]; ys = [p[1] for p in s["points"]]
            wh.append((max(xs)-min(xs), max(ys)-min(ys)))
        if n and os.path.exists(t.replace(".jpg", ".json")):
            with open(t.replace(".jpg", ".json")) as f:
                dt = json.load(f)
            if dt["shapes"]:
                cr = [sum(p[0] for p in s["points"])/4 for s in d["shapes"]][0] / d["imageWidth"]
                rr = [sum(p[1] for p in s["points"])/4 for s in d["shapes"]][0] / d["imageHeight"]
                ct = [sum(p[0] for p in s["points"])/4 for s in dt["shapes"]][0] / dt["imageWidth"]
                rt = [sum(p[1] for p in s["points"])/4 for s in dt["shapes"]][0] / dt["imageHeight"]
                center_delta.append((ct-cr, rt-rr))
print("per_scene images:", per_scene, "total", sum(per_scene.values()))
print("sampled pairs:", tot, "missing thermal:", missing)
print("labels:", dict(lab))
print("n_obj per image (rgb):", sorted(nobj.items())[:8])
wh = np.array(wh)
print("box w,h px  min %s  p10 %s  median %s  p90 %s  max %s" % (
    np.min(wh,0), np.percentile(wh,10,axis=0), np.median(wh,0),
    np.percentile(wh,90,axis=0), np.max(wh,0)))
area = wh[:,0]*wh[:,1]
print("box area px: median %.0f  p10 %.0f  p90 %.0f" % (np.median(area), np.percentile(area,10), np.percentile(area,90)))
print("rel area (frac of 1920x1080): median %.2e" % np.median(area/(1920*1080)))
cd = np.array(center_delta)
print("normalised center offset RGB->TIR: mean %s  std %s  p95|.| %s" % (
    cd.mean(0), cd.std(0), np.percentile(np.abs(cd),95,axis=0)))
print("n sequences (sampled):", len(seq), "sample seq ids:", list(seq)[:5])
