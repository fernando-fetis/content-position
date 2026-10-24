'''04_planes: removing the rotation without training, at three granularities. The planes each rule keeps (partial RoPE), and the KV heads and layers each rule derotates entirely, come from 01_measure's calibration; the exact edit is applied, and the rules are scored by the increase of the loss on held-out text at two lengths and by the tasks at one fraction. Results: <model>__selection (planes; read by 05_finetune), <model>__groups (KV heads and layers), <model>__intact, <model>__<rule> (planes), <model>__heads_<rule>, <model>__layers_<rule>, <model>__layer_<l>.'''
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import counterfactual as cf
from src.analysis.calibration import head_table, kv, windows
from src.analysis.plane_scores import kept, order_score
from src.core.models import Model, for_each_model
from src.edits.partial_rope import check_partial_rope, partial_rope
from src.eval.text_tasks import evaluate, evaluation_inputs
from src.utils.results import EXPERIMENTS, figures_dir, per_model, read, result_path, save, write_summary
from src.utils.stats import bootstrap_mean, spearman

SCORED = ('norm', 'mkl', 'diagonal', 'bound', 'variance')  # rules with a per-plane score in the calibration; EliteKV's comes from its order


def selection(cfg: dict, m: Model) -> dict:
    '''The masks kept[<rule>_<fraction>] (layers, KV heads, F), theta, the per-plane scores of every scored rule and EliteKV (variance_<T>: calibrated at length T), and seconds, the cost of the scores in each calibration pass.'''
    lengths = [cfg['calibration_tokens'], *cfg['long_calibration_tokens']]
    cals = {t: read(EXPERIMENTS / cfg['calibration_experiment'], m.name, f'calibration_{t}') for t in lengths}
    if missing := [t for t, c in cals.items() if c is None]:
        raise SystemExit(f'{m.name}: no calibration_{missing[0]} in {cfg["calibration_experiment"]}; run it first')
    cal, F = cals[cfg['calibration_tokens']], m.geometry.n_freqs
    scores = {**{k: cal['scores'][k] for k in SCORED}, **{f'variance_{t}': cals[t]['scores']['variance'] for t in cfg['long_calibration_tokens']}}
    masks = {f'{r}_{f}': kept({**scores, 'elitekv_order': cal['elitekv_order']}, r, max(1, round(f * F)), F) for r in cfg['rules'] for f in cfg['fractions']}
    return {'kept': masks, 'theta': m.geometry.theta().numpy(), 'scores': {**scores, 'elitekv': order_score(cal['elitekv_order'])}, 'seconds': {str(t): c['seconds'] for t, c in cals.items()}}


def head_count(f: float, n: int) -> int:
    return max(1, round(f * n))


def chosen_heads(scores: dict[str, np.ndarray], rule: str, f: float, seed: int) -> np.ndarray:
    '''The flat KV heads (layer * n_kv + head) a rule derotates at fraction f: the lowest scores, or for random_<i> a draw fixed by (seed, i, f).'''
    n = next(iter(scores.values())).size
    k = head_count(f, n)
    if rule.startswith('random'):
        return np.sort(np.random.default_rng([seed, int(rule.split('_')[1]), int(f * 100)]).choice(n, k, replace=False))
    return np.sort(np.argsort(scores[rule], kind='stable')[:k])


def chosen_layers(layer_variance: np.ndarray, rule: str, f: float, seed: int) -> np.ndarray:
    '''The layers a rule derotates at fraction f: variance, the lowest summed position-dependent variance; interleaved, every round(1 / f)-th layer; first, last; random_<i>, a draw fixed by (seed, i).'''
    L = layer_variance.size
    k = head_count(f, L)
    if rule == 'variance':
        return np.sort(np.argsort(layer_variance, kind='stable')[:k])
    if rule == 'interleaved':
        step = max(1, round(1 / f))
        return np.array([l for l in range(L) if (l + 1) % step == 0][:k], int)
    if rule == 'first':
        return np.arange(k)
    if rule == 'last':
        return np.arange(L - k, L)
    return np.sort(np.random.default_rng([seed, int(rule.split('_')[1]), 1000]).choice(L, k, replace=False))


def groups(cfg: dict, name: str) -> dict:
    '''From the calibration of 01 at calibration_tokens: per KV head the position-dependent variance Var(p_j) + Var(eps_ij) (averaged over calibration queries, summed over the query heads of the group; `cor:heads-without-rotation`) and the positional ratio (the largest over the non-sink query heads; all-sink groups first), per layer the variance summed over its heads; the KV heads (heads.<rule>_<f>, flat) and layers (layers.<rule>) each rule derotates.'''
    cal = read(EXPERIMENTS / cfg['calibration_experiment'], name, f'calibration_{cfg["calibration_tokens"]}')
    if cal is None:
        raise SystemExit(f'{name}: no calibration_{cfg["calibration_tokens"]} in {cfg["calibration_experiment"]}; run it first')
    n_kv = int(cal['n_kv'])
    var = (np.asarray(cal['var_p'], float) + np.asarray(cal['var_e'], float)).mean(0)  # (L, H)
    L, H = var.shape
    h = head_table(cal)
    head_variance = var.reshape(L, n_kv, H // n_kv).sum(-1).ravel()
    head_ph = kv(h['ratio'], n_kv, 'max', h['sink'] > cf.SINK, empty=-np.inf)
    layer_variance = var.sum(-1)
    scores = {'variance': head_variance, 'ph': head_ph, 'high_variance': -head_variance}
    heads = {f'{r}_{f}': chosen_heads(scores, r, f, cfg['seed']) for r in cfg['head_rules'] for f in cfg['head_fractions']}
    layers = {r: chosen_layers(layer_variance, r, cfg['layer_fraction'], cfg['seed']) for r in cfg['layer_rules']}
    return {'heads': heads, 'layers': layers, 'head_variance': head_variance, 'head_ph': head_ph, 'layer_variance': layer_variance, 'n_layers': L, 'n_kv': n_kv, 'n_freqs': len(cal['theta'])}


def group_mask(grp: dict, heads: np.ndarray | None = None, layers: np.ndarray | None = None) -> np.ndarray:
    '''Kept planes (layers, KV heads, F): every plane rotates except all those of the given flat KV heads or of every KV head of the given layers.'''
    kept = np.ones((int(grp['n_layers']), int(grp['n_kv']), int(grp['n_freqs'])), bool)
    if heads is not None:
        kept.reshape(-1, kept.shape[-1])[np.asarray(heads, int)] = False
    if layers is not None:
        kept[np.asarray(layers, int)] = False
    return kept


def compute(cfg: dict, exp: Path) -> None:
    for n in cfg['models']:
        save(result_path(exp, n, 'groups'), lambda n=n: groups(cfg, n))

    def run(m: Model) -> None:
        save(result_path(exp, m.name, 'selection'), lambda: selection(cfg, m))
        masks = read(exp, m.name, 'selection')['kept']
        inputs = evaluation_inputs(m, cfg)
        save(result_path(exp, m.name, 'intact'), lambda: evaluate(m, inputs, True))

        def edited(rule: str) -> dict:
            out = {}
            for f in cfg['fractions']:
                with partial_rope(m, np.asarray(masks[f'{rule}_{f}'], bool), per_model(cfg, 'chunk', m.name)):
                    out[str(f)] = evaluate(m, inputs, f == cfg['fraction'])
            return out

        for rule in cfg['rules']:
            save(result_path(exp, m.name, rule), lambda rule=rule: edited(rule))

        grp = read(exp, m.name, 'groups')
        if int(grp['n_freqs']) != m.geometry.n_freqs or int(grp['n_kv']) != m.geometry.n_kv_heads:
            raise SystemExit(f'{m.name}: the KV heads and layers of 01 do not match the model geometry')
        chunk = per_model(cfg, 'chunk', m.name)

        def derotated(kept: np.ndarray, tasks: bool, ins: dict = inputs) -> dict:
            with partial_rope(m, kept, chunk):
                return evaluate(m, ins, tasks)

        for r in cfg['head_rules']:
            save(result_path(exp, m.name, f'heads_{r}'), lambda r=r: {str(f): derotated(group_mask(grp, heads=grp['heads'][f'{r}_{f}']), f == cfg['fraction']) for f in cfg['head_fractions']})
        for r in cfg['layer_rules']:
            save(result_path(exp, m.name, f'layers_{r}'), lambda r=r: derotated(group_mask(grp, layers=grp['layers'][r]), True))
        if cfg['single_layers']:
            short = {'text': {k: v for k, v in inputs['text'].items() if k[1] == min(cfg['text_tokens'])}, 'tasks': {}}
            for l in range(int(grp['n_layers'])):
                save(result_path(exp, m.name, f'layer_{l}'), lambda l=l: derotated(group_mask(grp, layers=np.array([l])), False, short))

    for_each_model(cfg, cfg['models'], run)


def sigma_zeta2(theta: np.ndarray, tokens: int) -> np.ndarray:
    '''sigma_zeta^2 = 1 - delta_N(theta_f)^2 per plane (`prop:phase-statistics`), with delta_N the normalized Dirichlet kernel and N := T - 1 = tokens - 2 for a window of tokens tokens with the query last, as in counterfactual.parts.'''
    N, x = tokens - 2, np.asarray(theta, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        delta = np.where(x > 0, np.sin(N * x / 2) / (N * np.sin(x / 2)), 1.0)
    return 1 - delta ** 2


def model_summary(cfg: dict, exp: Path, n: str) -> dict | None:
    '''Per model: loss_increase, the increase of the loss (edited minus intact NLL, nats per token) per corpus and length, fraction and rule; intervals, paired 95% BCa intervals over windows for the pairs of compared; tasks, the accuracy per task at fraction; score_correlation.<rule>, the Spearman correlation between the plane scores of variance and of each other rule; sigma_zeta2.<T>.<rule>.share_below, at the shortest length per rule and at the other lengths for norm only (norm_to_variance reads them), the share of the planes kept at fraction whose sigma_zeta^2 is below norm_slow.threshold; overlap, the share of the planes variance keeps at fraction that each long calibration also keeps; seconds, the wall-clock cost of the closed-form parts and of EliteKV's order in the calibration pass at calibration_tokens, and their ratio. None until every rule is done.'''
    sel, intact = read(exp, n, 'selection'), read(exp, n, 'intact')
    if sel is None or intact is None:
        return None
    res = {r: d for r in cfg['rules'] if (d := read(exp, n, r)) is not None}
    if len(res) < len(cfg['rules']):
        return None
    increase = {(r, str(f), key): np.asarray(x, float) - np.asarray(intact['text'][key], float) for r, d in res.items() for f in cfg['fractions'] for key, x in d[str(f)]['text'].items()}
    loss_increase, intervals = {}, {}
    for (r, f, key), x in increase.items():
        loss_increase.setdefault(key, {}).setdefault(f, {})[r] = float(x.mean())
        for a, b in cfg['compared']:
            if r == a and (b, f, key) in increase:
                diff = x - increase[(b, f, key)]
                intervals.setdefault(key, {}).setdefault(f, {})[f'{a} - {b}'] = {'mean': float(diff.mean()), 'ci': list(bootstrap_mean(diff, cfg['n_boot'], cfg['seed']))}
    accuracy = lambda x: float(np.mean(np.asarray(x) > 0))
    tf = str(cfg['fraction'])
    tasks = {t: {'intact': accuracy(x), **{r: accuracy(d[tf]['tasks'][t]) for r, d in res.items()}} for t, x in intact['tasks'].items()}
    S = {k: np.asarray(v, float).ravel() for k, v in sel['scores'].items()}
    corr = {b: spearman(S['variance'], S[b]) for b in S if b != 'variance'}
    below: dict = {}
    for T in cfg['text_tokens']:
        sz = sigma_zeta2(sel['theta'], T)
        for r in cfg['norm_slow']['rules'] if T == cfg['text_tokens'][0] else ['norm']:
            mask = np.asarray(sel['kept'][f'{r}_{cfg["fraction"]}'], bool)
            v = np.broadcast_to(sz, mask.shape)[mask]
            below.setdefault(str(T), {})[r] = {'share_below': float(np.mean(v < cfg['norm_slow']['threshold']))}
    keep = {r: np.asarray(sel['kept'][f'{r}_{cfg["fraction"]}'], bool) for r in ('variance', *(f'variance_{t}' for t in cfg['long_calibration_tokens']))}
    overlap = {f'variance - {b}': float((keep['variance'] & keep[b]).sum() / keep['variance'].sum()) for b in keep if b != 'variance'}
    sec = sel['seconds'][str(cfg['calibration_tokens'])]
    seconds = {'parts': float(sec['parts']), 'elitekv': float(sec['elitekv']), 'elitekv_over_parts': float(sec['elitekv'] / sec['parts'])}
    return {'loss_increase': loss_increase, 'intervals': intervals, 'tasks': tasks, 'score_correlation': corr, 'sigma_zeta2': below, 'overlap': overlap, 'seconds': seconds, **({'rotation': rot} if (rot := rotation_summary(cfg, exp, n, intact)) else {})}


def rotation_summary(cfg: dict, exp: Path, n: str, intact: dict) -> dict | None:
    '''Heads and layers without rotation, per model: heads.loss_increase.<corpus>_<tokens>.<fraction>.<rule> and layers.loss_increase.<corpus>_<tokens>.<rule>, the increase of the loss (edited minus intact NLL, nats per token); heads.intervals and layers.intervals, paired 95% BCa intervals over windows of variance minus each other rule; tasks.<task>.<plan>, the accuracy (margin above zero) of the head plans at fraction (heads_<rule>) and of the layer plans (layers_<rule>); single_layers, from the increase of the loss of each layer alone at the shortest length, its Spearman correlation over layers with the layer's summed variance and, for the variance and last rules, the sum of these increases over the layers the rule chose; score_correlation, the Spearman correlation over KV heads of the variance and the positional ratio (over the KV heads with a non-sink query head); all_sink, the number of KV heads whose query heads are all sinks (the positional-ratio rule takes them first), out of all, and the median percentile of their variance among the KV heads; layers.chosen.variance, the layers the variance rule chose. None until every head rule, layer rule and single layer is done.'''
    grp = read(exp, n, 'groups')
    if grp is None:
        return None
    heads = {r: d for r in cfg['head_rules'] if (d := read(exp, n, f'heads_{r}')) is not None}
    layers = {r: d for r in cfg['layer_rules'] if (d := read(exp, n, f'layers_{r}')) is not None}
    singles = [read(exp, n, f'layer_{l}') for l in range(int(grp['n_layers']))] if cfg['single_layers'] else []
    if len(heads) < len(cfg['head_rules']) or len(layers) < len(cfg['layer_rules']) or any(s is None for s in singles):
        return None
    base = {k: np.asarray(x, float) for k, x in intact['text'].items()}
    accuracy = lambda x: float(np.mean(np.asarray(x) > 0))

    def compare(results: dict[str, dict]) -> tuple[dict, dict]:
        loss_increase, intervals = {}, {}
        for r, text in results.items():
            for k, x in text.items():
                loss_increase.setdefault(k, {})[r] = float((np.asarray(x, float) - base[k]).mean())
        if 'variance' in results:
            for r, text in results.items():
                if r != 'variance':
                    for k, x in text.items():
                        diff = np.asarray(results['variance'][k], float) - np.asarray(x, float)
                        intervals.setdefault(k, {})[f'variance - {r}'] = {'mean': float(diff.mean()), 'ci': list(bootstrap_mean(diff, cfg['n_boot'], cfg['seed']))}
        return loss_increase, intervals

    tasks: dict = {}
    head_out: dict = {'loss_increase': {}, 'intervals': {}}
    for f in cfg['head_fractions']:
        loss_increase, intervals = compare({r: d[str(f)]['text'] for r, d in heads.items()})
        for k in loss_increase:
            head_out['loss_increase'].setdefault(k, {})[str(f)] = loss_increase[k]
            head_out['intervals'].setdefault(k, {})[str(f)] = intervals.get(k, {})
    for r, d in heads.items():
        for t, x in d[str(cfg['fraction'])].get('tasks', {}).items():
            tasks.setdefault(t, {})[f'heads_{r}'] = accuracy(x)
    layer_increase, layer_intervals = compare({r: d['text'] for r, d in layers.items()})
    for r, d in layers.items():
        for t, x in d.get('tasks', {}).items():
            tasks.setdefault(t, {})[f'layers_{r}'] = accuracy(x)
    single: dict = {}
    if singles:
        lv = np.asarray(grp['layer_variance'], float)
        for k in singles[0]['text']:
            increase = np.array([float((np.asarray(s['text'][k], float) - base[k]).mean()) for s in singles])
            single[k] = {'spearman_variance': spearman(lv, increase), 'summed': {r: float(increase[np.asarray(grp['layers'][r], int)].sum()) for r in ('variance', 'last')}}
    hv, hp = np.asarray(grp['head_variance'], float), np.asarray(grp['head_ph'], float)
    ok = np.isfinite(hp)
    return {'heads': head_out, 'layers': {'loss_increase': layer_increase, 'intervals': layer_intervals, 'chosen': {'variance': [int(x) for x in grp['layers']['variance']]}}, 'tasks': tasks, 'single_layers': single,
            'score_correlation': spearman(hv[ok], hp[ok]),
            'all_sink': {'kv_heads': int((~ok).sum()), 'of': int(ok.size), 'variance_percentile': float(np.median((np.argsort(np.argsort(hv)) / max(hv.size - 1, 1))[~ok])) if (~ok).any() else float('nan')}}


def norm_to_variance(cfg: dict, models: dict) -> dict:
    '''Per corpus: ratio, the increase of the loss of the norm rule over that of the variance rule at fraction, per model and length; per length, the Spearman correlation over models between the share of the norm rule's kept planes with sigma_zeta^2 below norm_slow.threshold and the log of that ratio; the number of models where the ratio shrinks from the shortest to the longest length.'''
    f, lengths = str(cfg['fraction']), [str(T) for T in cfg['text_tokens']]
    out = {}
    for c in cfg['text']:
        ratio = {n: {T: s['loss_increase'][f'{c}_{T}'][f]['norm'] / s['loss_increase'][f'{c}_{T}'][f]['variance'] for T in lengths} for n, s in models.items()}
        with np.errstate(invalid='ignore'):
            corr = {T: spearman(np.array([s['sigma_zeta2'][T]['norm']['share_below'] for s in models.values()]), np.log([r[T] for r in ratio.values()])) for T in lengths}
        out[c] = {'ratio': ratio, 'spearman_share_below_vs_log_ratio': corr, 'shrinks': sum(r[lengths[-1]] < r[lengths[0]] for r in ratio.values()), 'models': len(models)}
    return out


def summarize(cfg: dict, exp: Path) -> None:
    models = {n: s for n in cfg['models'] if (s := model_summary(cfg, exp, n)) is not None}
    write_summary(exp, {'models': models, 'norm_to_variance': norm_to_variance(cfg, models)})


def figures(cfg: dict, exp: Path) -> None:
    from figures import draw
    draw(cfg, read(exp, 'summary'), figures_dir(exp))


def check(cfg: dict, exp: Path) -> None:
    '''With every plane kept, both partial RoPEs give the intact logits within check.tolerance; when the selection exists, every mask keeps max(1, round(f F)) planes per KV head and each rule's masks are nested across fractions; when the KV heads and layers exist, every head plan derotates max(1, round(f n)) distinct KV heads, every layer plan at most max(1, round(f L)) distinct layers, and the masks remove exactly those.'''
    def run(m: Model) -> None:
        g = m.geometry
        ids = windows(m, cfg['text']['wiki'], 1, cfg['check']['tokens'], cfg['seed'] + 2)[0]
        dev = check_partial_rope(m, ids)
        print(f'{m.name}: partial RoPE with every plane kept deviates from the intact model by {dev:.2e} ({len(ids)} tokens)', flush=True)
        if dev > cfg['check']['tolerance']:
            raise SystemExit(f'{m.name}: check failed, deviation {dev:.2e} above {cfg["check"]["tolerance"]:g}')
        if (sel := read(exp, m.name, 'selection')) is None:
            print(f'{m.name}: no selection yet, masks not checked', flush=True)
            return
        for rule in cfg['rules']:
            previous = np.zeros((g.n_layers, g.n_kv_heads, g.n_freqs), bool)
            for f in cfg['fractions']:
                mask = np.asarray(sel['kept'][f'{rule}_{f}'], bool)
                if not (mask.sum(-1) == max(1, round(f * g.n_freqs))).all() or (previous & ~mask).any():
                    raise SystemExit(f'{m.name}: {rule}_{f} does not keep round(f F) planes per KV head, or drops planes kept at a smaller fraction')
                previous = mask
        print(f'{m.name}: {len(sel["kept"])} masks consistent', flush=True)
        if (grp := read(exp, m.name, 'groups')) is None:
            print(f'{m.name}: no KV heads and layers yet, plans not checked', flush=True)
            return
        n, L = g.n_layers * g.n_kv_heads, g.n_layers
        for key, idx in grp['heads'].items():
            idx = np.asarray(idx, int)
            k = head_count(float(key.rsplit('_', 1)[1]), n)
            if idx.size != k or np.unique(idx).size != k or not (group_mask(grp, heads=idx).reshape(n, -1).any(-1) == ~np.isin(np.arange(n), idx)).all():
                raise SystemExit(f'{m.name}: head plan {key} does not derotate {k} distinct KV heads')
        for key, idx in grp['layers'].items():
            idx = np.asarray(idx, int)
            if idx.size == 0 or idx.size > head_count(cfg['layer_fraction'], L) or np.unique(idx).size != idx.size or not (group_mask(grp, layers=idx).any((1, 2)) == ~np.isin(np.arange(L), idx)).all():
                raise SystemExit(f'{m.name}: layer plan {key} is not a set of at most round(f L) layers')
        print(f'{m.name}: {len(grp["heads"]) + len(grp["layers"])} head and layer plans consistent', flush=True)

    for_each_model(cfg, cfg['models'], run)
