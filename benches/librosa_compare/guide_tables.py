"""Write the result tables of the librosa guide from one run's records.

    guide_tables.py OUT_DIR DEST_DIR

Reads OUT_DIR: merged.json (report.py), load_*.jsonl (load_bench.py), startup*.jsonl
(startup_bench.py), footprint.json (footprint.py), versions.jsonl, and the WAV harness's
wav_rw.csv and wav_mem.csv. Writes one reStructuredText table per file into DEST_DIR and the
inline figures into DEST_DIR/substitutions.txt; the guide includes them. Rerun after a new
benchmark run; nothing in the tables is written by hand.
"""

import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

G = sys.argv[1]
DEST = sys.argv[2]
def J(n):
    p=os.path.join(G,n)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
rows=json.load(open(G+"/merged.json"))
vers=J("versions.jsonl")[-1]
def geo(x): return math.exp(sum(map(math.log,x))/len(x))
def table(head,body,widths=None):
    o=[".. list-table::","   :header-rows: 1"]
    if widths: o.append("   :widths: "+" ".join(map(str,widths)))
    o.append("")
    for r in [head]+body:
        for i,c in enumerate(r):
            o.append(("   * - " if i==0 else "     - ")+str(c))
    return "\n".join(o)+"\n"
def mem(x):
    return f"{x/1024:.0f} KB" if x<1024**2 else f"{x/1024**2:.1f} MB"
def ms(x): return f"{x:.3g}"
def cnt(a,b):
    lo=eq=hi=0
    for r in rows:
        x,y=r.get(a),r.get(b)
        if x is None or y is None: continue
        lo+=x<y; eq+=x==y; hi+=x>y
    return lo,eq,hi
OUT={}
ms_rows=[("Time through Python, one core","ours_py_ms","librosa_ms"),("Time, native Rust, one core","rust_ms","librosa_ms"),
("Time through Python, default threading","ours_default_ms","librosa_default_ms"),("Time, native Rust, default threading","rust_default_ms","librosa_default_ms"),
("Peak allocation per call through Python","ours_peak","librosa_peak"),("Peak allocation per call, native","rust_peak","librosa_peak"),("Peak resident growth per call","ours_rss_kb","librosa_rss_kb")]
OUT["COUNTS"]=table(["Measure","Rows","Ours lower","Equal","Ours higher"],[[n,sum(cnt(a,b)),*cnt(a,b)] for n,a,b in ms_rows],[46,10,14,10,14])
def R(rs,a,b): return sorted(r[a]/r[b] for r in rs if r.get(a) and r.get(b))
body=[]
names={"spectrum":"STFT, dB, mel, MFCC, windows","features":"Spectral features","tonal":"Chroma, tuning","constant_q":"Constant-Q","pitch":"YIN, pYIN, harmonics","rhythm":"Onsets, tempo, beats","effects":"Effects, HPSS, resampling","sequence":"DTW, Viterbi, recurrence","convert":"Unit conversions"}
for g in list(names)+[None]:
    rs=[r for r in rows if g is None or r["group"]==g]
    t,n,m=R(rs,"ours_py_ms","librosa_ms"),R(rs,"rust_ms","librosa_ms"),R(rs,"ours_peak","librosa_peak")
    body.append([names.get(g,"**All rows**"),len(rs),f"{st.median(t):.2f}",f"{t[-1]:.2f}",f"{st.median(n):.2f}",f"{st.median(m):.2f}",f"{m[-1]:.2f}"])
OUT["AREAS"]=table(["Area","Rows","Time, median","Time, worst","Native time, median","Memory, median","Memory, worst"],body,[34,8,10,10,12,12,10])
allt=R(rows,"ours_py_ms","librosa_ms"); allm=R(rows,"ours_peak","librosa_peak")
OUT["OVERALL"]=f"median time ratio {st.median(allt):.2f} (90th percentile {allt[int(.9*len(allt))]:.2f}, worst {allt[-1]:.3f}), median memory ratio {st.median(allm):.2f} (worst {allm[-1]:.2f})"
by=defaultdict(dict)
for r in rows: by[r["case"]][r["config"]]=r
common=["stft","istft","melspectrogram","mfcc","chroma_stft","chroma_cqt","cqt","spectral_centroid","rms","zero_crossing_rate","onset_strength","tempo","beat_track","yin","pyin","hpss","time_stretch","pitch_shift","griffinlim","trim","amplitude_to_db","dtw","viterbi","recurrence_matrix"]
body=[]
for c in common:
    for cfg in ("22050_10s_mono_f32","seq_f32"):
        r=by.get(c,{}).get(cfg)
        if r: body.append([f"``{c}``",ms(r["ours_py_ms"]),ms(r["rust_ms"]),ms(r["librosa_ms"]),f"{r['ratio_py']:.2f}",mem(r["ours_peak"]),mem(r["librosa_peak"])])
OUT["COMMON"]=table(["Function","Ours, Python ms","Ours, native ms","librosa ms","Time ratio","Ours peak","librosa peak"],body,[28,12,12,12,10,13,13])
weak=sorted((r for r in rows if r.get("ratio_py")),key=lambda r:-r["ratio_py"])[:10]
OUT["WEAK"]=table(["Case","Input","Ours ms","librosa ms","Ratio"],[[f"``{r['case']}``",r["config"],f"{r['ours_py_ms']:.4g}",f"{r['librosa_ms']:.4g}",f"{r['ratio_py']:.3f}"] for r in weak],[34,30,12,12,12])
# load
t={};m={};rr=defaultdict(list)
for r in J("load_time.jsonl"):
    if "median_ms" in r: t[(r["row"],r["input"],r["lib"])]=r
for r in J("load_mem.jsonl"):
    if "mr_peak" in r: m[(r["row"],r["input"],r["lib"])]=r
for r in J("load_rss.jsonl"):
    if "rss_delta_kb" in r: rr[(r["row"],r["input"],r["lib"])].append(r["rss_delta_kb"])
labels={"read_f32":("Whole file as float32","``read(p, as_type=F32)``","``load(p, sr=None, mono=False)``"),
"read_i16":("Whole file as int16","``read(p, as_type=I16)``","``load(p, sr=None, mono=False, dtype=int16)``"),
"read_mono_f32":("Mono float32","``read(p, as_type=F32).to_mono()``","``load(p, sr=None)``"),
"load_default":("Mono float32 at 22050 Hz","``read(...).to_mono().resample(22050, \"high\")``","``load(p)``"),
"read_resample_stereo":("Stereo float32 at 22050 Hz","``read_and_resample(p, 22050)``","``load(p, mono=False)``"),
"read_part":("2 s from 5 s in","``StreamedAudioReader`` seek and read","``load(p, sr=None, mono=False, offset=5, duration=2)``"),
"stream_blocks_stereo":("Stream, overlapping blocks","``windows(...)``","``stream(...)``"),
"stream_chunks_stereo":("Stream, 4096 frames","``frames(4096)``","``stream(...)``")}
body=[]
for key,(lab,ours,theirs) in labels.items():
    for inp,inpl in (("wav16_60s","WAV"),("flac16_60s","FLAC")):
        a,b=t.get((key,inp,"ours")),t.get((key,inp,"librosa"))
        ma,mb=m.get((key,inp,"ours")),m.get((key,inp,"librosa"))
        if not a or not b: continue
        body.append([lab,inpl,ms(a["median_ms"]),ms(b["median_ms"]),f"{a['median_ms']/b['median_ms']:.2f}",mem(ma["mr_peak"]) if ma else "",mem(mb["mr_peak"]) if mb else ""])
OUT["LOAD"]=table(["Work (60 s, stereo, 16-bit, 44100 Hz)","File","Ours ms","librosa ms","Ratio","Ours peak","librosa peak"],body,[40,8,10,10,8,12,12])
cnts=dict(time=[0,0,0],mem=[0,0,0],rss=[0,0,0]); lose=[]
for k in sorted({k[:2] for k in t}):
    if k[0]=="defaults": continue
    a,b=t.get((*k,"ours")),t.get((*k,"librosa"))
    if a and b:
        cnts["time"][0 if a["median_ms"]<b["median_ms"] else 2]+=1
        if a["median_ms"]>=b["median_ms"]: lose.append(("time",k,a["median_ms"],b["median_ms"]))
    a,b=m.get((*k,"ours")),m.get((*k,"librosa"))
    if a and b:
        cnts["mem"][0 if a["mr_peak"]<b["mr_peak"] else 1 if a["mr_peak"]==b["mr_peak"] else 2]+=1
        if a["mr_peak"]>b["mr_peak"]: lose.append(("mem",k,a["mr_peak"],b["mr_peak"]))
    a,b=rr.get((*k,"ours")),rr.get((*k,"librosa"))
    if a and b:
        x,y=st.median(a),st.median(b)
        cnts["rss"][0 if x<y else 1 if x==y else 2]+=1
        if x>y: lose.append(("rss",k,x,y))
OUT["LOADCOUNTS"]=f"time {cnts['time'][0]} lower, {cnts['time'][2]} higher; peak allocation {cnts['mem'][0]} lower, {cnts['mem'][1]} equal, {cnts['mem'][2]} higher; peak resident growth {cnts['rss'][0]} lower, {cnts['rss'][1]} equal, {cnts['rss'][2]} higher"
near=[(k[:2],round(t[(*k[:2],"ours")]["median_ms"]/t[(*k[:2],"librosa")]["median_ms"],3)) for k in t if k[2]=="ours" and (*k[:2],"librosa") in t and t[(*k[:2],"ours")]["median_ms"]/t[(*k[:2],"librosa")]["median_ms"]>=0.97]
# start-up
S=J("startup.jsonl"); SD=J("startup_default.jsonl")
imp={r["name"]:r for r in S if r["stage"]=="import"}
def rg(r): return f"{r['median_ms']:.1f} ({r['min_ms']:.1f}-{r['max_ms']:.1f})"
OUT["IMPORT"]=table(["``python -c``","Median ms (range), 30 fresh processes"],[[f"``{c}``",rg(imp[k])] for k,c in (("python","pass"),("audio_samples","import audio_samples"),("librosa","import librosa"),("numpy","import numpy"),("numpy_audio_samples","import numpy, audio_samples"),("numpy_librosa","import numpy, librosa"),("librosa_feature","import librosa; librosa.feature.mfcc"))],[60,40])
fc=defaultdict(dict)
for r in S:
    if r["stage"]=="first": fc[r["name"]][r["lib"]]=r
OUT["FIRST"]=table(["Function","Ours, first call","Ours, third call","librosa, first call","librosa, third call"],[[f"``{k}``",rg(v["ours"]),ms(v["ours"]["third_ms"]),rg(v["librosa"]),ms(v["librosa"]["third_ms"])] for k,v in fc.items() if k in("stft","mfcc","chroma_stft","cqt","onset_detect","beat_track","yin","pyin","hpss","time_stretch","resample","griffinlim","dtw","viterbi","recurrence_matrix")],[24,22,12,28,14])
d=defaultdict(dict)
for r in S:
    if r["stage"]=="mfcc": d[(r["input"],r["name"])][r["lib"]]=r
OUT["MFCCPROC"]=table(["File (stereo, 44100 Hz)","Settings","Ours ms (range)","librosa ms (range)"],[[k[0].split(".")[0].replace("wav16_","16-bit WAV, ").replace("flac16_","16-bit FLAC, "),{"mono_22050":"mono, 22050 Hz","mono_native":"mono, 44100 Hz"}[k[1]],rg(v["ours"]),rg(v["librosa"])] for k,v in d.items()],[34,22,22,22])
b=[r for r in SD if r["stage"]=="batch"]
th=["1","2","4","8","16"]
OUT["BATCH"]=table(["16 files of 60 s","Library"]+[f"{x} thread{'s' if x!='1' else ''}" for x in th],[[r["kind"].upper(),{"ours":"ours","librosa":"librosa"}[r["lib"]]]+[f"{r['ms_by_threads'][x]:.0f}" for x in th] for r in b],[20,14,11,11,11,11,11])
it=[r for r in S if r["stage"]=="interop"]
if it:
    u=it[-1]["us"]
    OUT["INTEROP"]=table(["Operation","Microseconds (median)"],[[k,f"{v:.2f}" if v>=1 else f"{v:.3f}"] for k,v in u.items() if not k.startswith("melspectrogram")],[70,30])
# WAV
def read_csv(name):
    p = os.path.join(G, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
d = {}
for r in read_csv("wav_rw.csv"):
    if r["p50_ms"] and r["p50_ms"] != "nan":
        d[(r["lib"], r["condition"], r["dtype"], r["channels"], r["duration_s"])] = float(r["p50_ms"])
if d:
    body = []
    for lib in ("librosa", "soundfile", "scipy", "wave"):
        for cond in ("bulk_read", "bulk_write"):
            x = [a / v for k, v in d.items() if k[0] == lib and k[1] == cond for a in [d.get(("audio_samples",) + k[1:])] if a]
            if x:
                body.append([f"``{lib}``" + (" (``load``)" if lib == "librosa" else ""), cond[5:], len(x), sum(q > 1 for q in x), f"{st.median(x):.2f}", f"{max(x):.3f}"])
    OUT["WAV"] = table(["Against", "Operation", "Rows", "Ours slower", "Median ratio", "Worst ratio"], body, [30, 14, 10, 14, 16, 16])
mem_rows = read_csv("wav_mem.csv")
if mem_rows:
    pk = {(r["lib"], r["condition"], r["dtype"], r["channels"], r["duration_s"]): int(r["mr_peak"]) for r in mem_rows}
    higher = [k for k in pk if k[0] != "audio_samples" and pk.get(("audio_samples",) + k[1:], 0) > pk[k]]
    OUT["WAVMEM"] = f"{sum(1 for k in pk if k[0] != 'audio_samples') - len(higher)} of {sum(1 for k in pk if k[0] != 'audio_samples')}"
f = json.load(open(os.path.join(G, "footprint.json"))) if os.path.exists(os.path.join(G, "footprint.json")) else None
if f:
    MB = lambda b: f"{b / 1e6:.0f} MB"
    OUT["FOOTPRINT"] = table(["", "Ours", "librosa 1.0.0"], [
        ["Own wheel", f"{f['ours_wheel']['bytes'] / 1e6:.1f} MB", f"{f['librosa_wheel']['bytes'] / 1e6:.1f} MB"],
        ["Fresh environment, with dependencies", MB(f["ours"]["site_packages_bytes"]), MB(f["librosa"]["site_packages_bytes"])],
        ["Packages installed", f["ours"]["package_count"], f["librosa"]["package_count"]],
    ], [50, 25, 25])
# FLAC encoding at matched levels, from the criterion results of flac_rust.json.
p = os.path.join(G, "flac_rust.json")
if os.path.exists(p):
    fr = json.load(open(p))
    OUT["FLACWRITE"] = table(["Compression level", "Ours / libFLAC, median", "Lowest", "Highest"],
                             [[r["level"], f"{r['median']:.2f}", f"{r['min']:.2f}", f"{r['max']:.2f}"] for r in fr], [40, 24, 18, 18])
OUT["WHEN"]=vers["when"][:10]
OUT["RUST"]=vers["rust_commit"]; OUT["PY"]=vers["python_commit"]
os.makedirs(DEST, exist_ok=True)
with open(os.path.join(DEST, "substitutions.txt"), "w") as fh:
    for k, v in OUT.items():
        if "\n" not in v:
            fh.write(f".. |{k.lower()}| replace:: {v}\n")
for k, v in OUT.items():
    if "\n" in v:
        with open(os.path.join(DEST, f"{k.lower()}.txt"), "w") as fh:
            fh.write(v)
print("wrote", len(OUT), "items to", DEST)
for label, x in (("load rows above librosa", lose), ("load near ties", near)):
    print(label, x)
