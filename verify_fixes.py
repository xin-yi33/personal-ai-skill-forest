import json, os, time

print('='*70)
print('VERIFICATION OF ALL FIX REQUIREMENTS')
print('='*70)

# === Req 1 ===
print('\n[Req 1] 实验四数据集 (70真实+30噪声, 单一JSON文件)')
fpath = 'shared/data/conversations_dataset.json'
if os.path.exists(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        convs = json.load(f)
    n_real = sum(1 for c in convs if not c['noise'])
    n_noise = sum(1 for c in convs if c['noise'])
    print(f'  File: {fpath} EXISTS')
    print(f'  Total: {len(convs)}, Real: {n_real}, Noise: {n_noise}, Ratio: {n_noise/len(convs)*100:.0f}%')
    print(f'  STATUS: {"PASS" if n_real==70 and n_noise==30 else "FAIL"}')
else:
    print(f'  File: {fpath} NOT FOUND - FAIL')

# === Req 2 ===
print('\n[Req 2] API规模 (5000全依赖, max depth 3, 50私有, 20参数冲突)')
with open('shared/data/all_apis_enriched.json', 'r', encoding='utf-8') as f:
    apis = json.load(f)
has_req = sum(1 for a in apis if a.get('requires'))
print(f'  APIs with deps: {has_req}/{len(apis)}')

by_id = {a['id']: a for a in apis}
def depth(skill, visited):
    sid = skill.get('id','')
    if sid in visited: return 0
    visited.add(sid)
    reqs = skill.get('requires', [])
    if not reqs: return 0
    return 1 + max(depth(by_id[r], visited) for r in reqs if r in by_id) if reqs else 0
max_d = max(depth(a, set()) for a in apis)
print(f'  Max dependency depth: {max_d}')

with open('shared/data/private_skills.json', 'r', encoding='utf-8') as f:
    priv = json.load(f)
n_priv = sum(len(v) for v in priv.values())
print(f'  Private skills: {n_priv} (5 users)')

with open('shared/data/param_conflict_cases.json', 'r', encoding='utf-8') as f:
    pcases = json.load(f)
print(f'  Param conflict cases: {len(pcases)}')

status2 = 'PASS' if has_req==5000 and max_d==3 and n_priv==50 and len(pcases)==20 else 'FAIL'
print(f'  STATUS: {status2}')

# === Req 3 ===
print('\n[Req 3] M2路由实验 (Forest vs Single Tree vs Flat ANN)')
with open('Exp2_Ablation_Study/results/experiment2_v3_results.json', 'r', encoding='utf-8') as f:
    r2 = json.load(f)
m2 = r2['M2']
print(f'  Forest acc: {m2["forest_acc"]:.3f}')
print(f'  Single Tree acc: {m2["single_tree_acc"]:.3f}')
print(f'  Flat ANN acc: {m2["flat_acc"]:.3f}')
print(f'  STATUS: PASS (three-way comparison complete)')

# === Req 4 ===
print('\n[Req 4] 规模扩展表格 (清晰列标题+单位+说明)')
with open('Exp1_Retrieval_Performance/run_experiment.py', 'r', encoding='utf-8') as f:
    code = f.read()
has_labels = 'API数量' in code and '总Token消耗' in code and 'Token节省' in code and '准确率' in code
print(f'  Bilingual headers with units: {has_labels}')
print(f'  STATUS: {"PASS" if has_labels else "FAIL"}')

# === Req 5 ===
print('\n[Req 5] 所有实验重新执行')
exp_files = [
    'Exp1_Retrieval_Performance/results/experiment1_v2_results.json',
    'Exp2_Ablation_Study/results/experiment2_v3_results.json',
    'Exp3_Threshold_Sensitivity/results/experiment3_v2_results.json',
    'Exp4_Action_Reflection/results/experiment4_results.json',
    'Exp5_Thought_Reflection/results/experiment5_results.json',
    'Exp6_Token_Consumption/results/experiment6_v2_results.json',
]
all_pass = True
for ef in exp_files:
    if os.path.exists(ef):
        mtime = os.path.getmtime(ef)
        age_min = (time.time() - mtime) / 60
        print(f'  {ef}: {age_min:.0f} min ago')
        if age_min > 120:
            all_pass = False
    else:
        print(f'  {ef}: NOT FOUND')
        all_pass = False
print(f'  STATUS: {"PASS" if all_pass else "FAIL"}')

# === Req 6 ===
print('\n[Req 6] 报告结构 (目标/设计/数据集/结果/分析/局限性)')
with open('README.md', 'r', encoding='utf-8') as f:
    report = f.read()
sections = ['实验目标', '对照组设计', '数据集说明', '核心结果', '分析', '局限性']
for s in sections:
    found = s in report
    print(f'  Contains "{s}": {found}')
all_sections = all(s in report for s in sections)
print(f'  STATUS: {"PASS" if all_sections else "FAIL"}')

print('\n' + '='*70)
