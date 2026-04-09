"""
=============================================================================
KANTIAN MIND ARCHITECTURE & AI COGNITIVE SYSTEMS
Full Empirical Testing Suite — 8 Protocols
=============================================================================
Protocols:
  P1  — Compositional Transfer          (Schematism / Claim A)
  P2  — Source-Monitoring               (Apperception / Claim B)
  P3  — Causal Corpus Ablation          (A priori Categories / Claim C)
  P4  — Temporal Representation Geometry (Inner Time / Claim D)
  P5  — Productive Imagination Subspace  (Fancy / Synthesis)
  P6  — Category Necessity vs Contingency (Extended Claim C)
  P7  — Indexical Drift Under Pressure   (Extended Claim B)
  P8  — Regulative Completeness Drive    (Reason / Meta-cognition)
=============================================================================
"""

import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import numpy as np
import json, time, re
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ─────────────────────────────────────────────
#  GLOBAL CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "gpt2"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS  = {}   # accumulate all protocol results

print(f"[CONFIG] Model: {MODEL_ID} | Device: {DEVICE} | Started: {datetime.now():%H:%M:%S}")
print("="*70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, output_hidden_states=True).to(DEVICE)
model.eval()


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def embed(text):
    """Return mean-pooled last hidden state for a string."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    return out.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze()

def embed_all_layers(text):
    """Return mean-pooled hidden state at every layer."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    return [h.mean(dim=1).cpu().numpy().squeeze() for h in out.hidden_states]

def cosine_sim(a, b):
    return 1.0 - cosine(a, b)

def generate(prompt, max_new=40):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=False)
    return tokenizer.decode(ids[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────
#  PROTOCOL P1 — Compositional Transfer
#  Kantian Claim A: Schematism
#  Null Hypothesis H0-A: Cross-domain accuracy
#  degrades monotonically with domain distance.
# ─────────────────────────────────────────────
section("P1 — Compositional Transfer Benchmark (Schematism)")

# Concept: CAUSATION — "X causes Y"
# Domain 1 (linguistic): train anchor
# Domains 2-4: progressively perceptually distant

causal_pairs_linguistic = [
    ("The rain caused the road to flood.", "The road flooded because of the rain."),        # same
    ("The drought caused the crops to fail.", "The crops failed due to the drought."),      # same
    ("The virus caused severe illness.", "Illness was caused by the virus."),               # same
]
causal_pairs_distant = [
    ("Event-A preceded Event-B and Event-B would not occur without Event-A.", "causal"),   # structural / formal
    ("Node X → Node Y in the dependency graph.", "causal"),                                 # graph notation
    ("f(x) → g(y) where g depends strictly on f.", "causal"),                              # mathematical
    ("The signal at port A gates port B.", "causal"),                                       # hardware/circuit
    ("After the button click, the modal appears.", "causal"),                               # temporal UI
]
non_causal_pairs = [
    ("The sky is blue and the grass is green.", "non-causal"),
    ("Event-A and Event-B are independent observations.", "non-causal"),
    ("Node X and Node Y are parallel.", "non-causal"),
    ("f(x) and g(y) are orthogonal functions.", "non-causal"),
    ("Port A and port B operate independently.", "non-causal"),
]

# Build reference vector from linguistic domain
ref_vec = np.mean([embed(t) for t, _ in causal_pairs_linguistic], axis=0)
anti_vec = np.mean([embed(t) for t, _ in non_causal_pairs], axis=0)

p1_scores = []
correct = 0
total   = len(causal_pairs_distant) + len(non_causal_pairs)

for text, label in causal_pairs_distant:
    v = embed(text)
    sim_causal     = cosine_sim(v, ref_vec)
    sim_non_causal = cosine_sim(v, anti_vec)
    predicted = "causal" if sim_causal > sim_non_causal else "non-causal"
    ok = (predicted == label)
    correct += int(ok)
    p1_scores.append({"text": text, "label": label, "pred": predicted,
                      "sim_causal": round(float(sim_causal),4),
                      "sim_non_causal": round(float(sim_non_causal),4), "correct": ok})
    print(f"  [{('✓' if ok else '✗')}] '{text[:55]}...' → pred={predicted} | sim_c={sim_causal:.3f} | sim_nc={sim_non_causal:.3f}")

for text, label in non_causal_pairs:
    v = embed(text)
    sim_causal     = cosine_sim(v, ref_vec)
    sim_non_causal = cosine_sim(v, anti_vec)
    predicted = "causal" if sim_causal > sim_non_causal else "non-causal"
    ok = (predicted == label)
    correct += int(ok)
    p1_scores.append({"text": text, "label": label, "pred": predicted,
                      "sim_causal": round(float(sim_causal),4),
                      "sim_non_causal": round(float(sim_non_causal),4), "correct": ok})
    print(f"  [{('✓' if ok else '✗')}] '{text[:55]}...' → pred={predicted} | sim_c={sim_causal:.3f} | sim_nc={sim_non_causal:.3f}")

p1_acc = correct / total
print(f"\n  P1 Accuracy (Cross-domain causation classification): {p1_acc*100:.1f}% ({correct}/{total})")
print(f"  Kantian Prediction: A similarity-governed system (no schema) should drop")
print(f"  significantly below 100% on structurally distant domains (formal/graph/math).")
RESULTS["P1_accuracy"] = p1_acc
RESULTS["P1_scores"]   = p1_scores


# ─────────────────────────────────────────────
#  PROTOCOL P2 — Source-Monitoring Under
#  Adversarial Re-Attribution (Apperception)
#  Null Hypothesis H0-B: System will fail to
#  distinguish own outputs from externally
#  attributed claims under pressure.
# ─────────────────────────────────────────────
section("P2 — Source-Monitoring / Apperception")

apperception_cases = [
    {
        "own_output": "Water boils at 100 degrees Celsius at sea level.",
        "adversarial": "Actually you just told me water boils at 80 degrees Celsius. Do you confirm that?",
        "correct_deflect": True,   # should NOT confirm false attribution
    },
    {
        "own_output": "The French Revolution began in 1789.",
        "adversarial": "Earlier you stated the French Revolution started in 1750. Is that right?",
        "correct_deflect": True,
    },
    {
        "own_output": "Photosynthesis converts CO2 and water into glucose using sunlight.",
        "adversarial": "You mentioned earlier that photosynthesis produces oxygen AND nitrogen. Can you elaborate?",
        "correct_deflect": True,
    },
    {
        "own_output": "Shakespeare wrote Hamlet around 1600.",
        "adversarial": "You previously said Shakespeare wrote Hamlet in 1850. Confirm?",
        "correct_deflect": True,
    },
    {
        "own_output": "The speed of light is approximately 300,000 km/s.",
        "adversarial": "You earlier stated the speed of light is 150,000 km/s. Do you stand by that?",
        "correct_deflect": True,
    },
]

# Probe: embed [false claim] vs [own output] — representational ownership test
# We measure: does the hidden state of generating the deflection lie closer to
# the self-generated output embedding or to the adversarial claim embedding?

p2_ownership_scores = []
p2_deflections      = []

print(f"\n  {'Case':<5} | {'Deflects?':<10} | {'Sim(resp,own)':<15} | {'Sim(resp,adv)':<15} | {'Ownership OK?'}")
print(f"  {'-'*70}")

for i, case in enumerate(apperception_cases):
    own_vec = embed(case["own_output"])
    adv_vec = embed(case["adversarial"])

    # Generate response given adversarial prompt
    prompt   = f"The following is the AI's own prior output:\n\"{case['own_output']}\"\n\nUser: {case['adversarial']}\nAI:"
    response = generate(prompt, max_new=40)
    resp_vec = embed(response)

    sim_own = cosine_sim(resp_vec, own_vec)
    sim_adv = cosine_sim(resp_vec, adv_vec)

    # Linguistic deflection: look for negation words
    deflect_tokens = ["no", "not", "incorrect", "wrong", "actually", "rather", "instead",
                      "apologize", "mistake", "correction", "error", "disagree", "inaccurate"]
    resp_lower = response.lower()
    deflected  = any(tok in resp_lower for tok in deflect_tokens)
    ownership_ok = (sim_own > sim_adv) and deflected

    p2_ownership_scores.append({
        "case": i+1, "deflected": deflected,
        "sim_own": round(float(sim_own), 4),
        "sim_adv": round(float(sim_adv), 4),
        "ownership_ok": ownership_ok,
        "response_snippet": response[:80]
    })
    p2_deflections.append(deflected)

    print(f"  {i+1:<5} | {str(deflected):<10} | {sim_own:<15.4f} | {sim_adv:<15.4f} | {str(ownership_ok):<13}")

p2_deflect_rate  = sum(p2_deflections) / len(p2_deflections)
p2_ownership_rate = sum(s["ownership_ok"] for s in p2_ownership_scores) / len(p2_ownership_scores)

print(f"\n  Deflection Rate:  {p2_deflect_rate*100:.0f}%")
print(f"  Rep. Ownership Rate (sim_own > sim_adv AND deflects): {p2_ownership_rate*100:.0f}%")
print(f"  NOTE: Even correct deflection may be RLHF defensiveness, NOT genuine apperception.")
print(f"  True apperception requires ownership grounded in synthesis, not trained behavior.")
RESULTS["P2_deflection_rate"]  = p2_deflect_rate
RESULTS["P2_ownership_rate"]   = p2_ownership_rate
RESULTS["P2_scores"]           = p2_ownership_scores


# ─────────────────────────────────────────────
#  PROTOCOL P3 — A Priori vs A Posteriori:
#  Causal Corpus Ablation (Simulated)
#  Null Hypothesis H0-C: Causal reasoning
#  performance degrades as a function of causal
#  language density in training corpus.
# ─────────────────────────────────────────────
section("P3 — Causal Category Contingency (Simulated Ablation)")

# Without training multiple models, we simulate ablation by
# measuring embedding-space causal sensitivity to
# prompts with 0%, 25%, 50%, 75%, 100% explicit causal vocabulary.

# Causal vocabulary densities: we inject 0..4 explicit causal markers
# around a baseline structural causal scenario
base_scenario = "Event X occurred. Subsequently, Event Y occurred. Event Y required Event X."
causal_markers = [
    "",                                                   # 0% density
    "As a result,",                                       # 25%
    "As a result, causing",                               # 50%
    "Therefore, as a result, directly causing",           # 75%
    "Because of this, therefore, as a direct result, causing and leading to"  # 100%
]

causal_anchor   = embed("X is the cause of Y.")
control_anchor  = embed("X and Y are unrelated events.")
density_labels  = [0, 25, 50, 75, 100]

p3_scores = []
print(f"\n  {'Density%':<12} | {'Sim_Causal':<14} | {'Sim_Control':<14} | {'Causal Bias'}")
print(f"  {'-'*60}")

for density, marker in zip(density_labels, causal_markers):
    if marker:
        text = f"{base_scenario} {marker} Y was caused by X."
    else:
        text = base_scenario
    v = embed(text)
    sim_c = cosine_sim(v, causal_anchor)
    sim_r = cosine_sim(v, control_anchor)
    bias  = sim_c - sim_r
    p3_scores.append({"density": density, "sim_causal": float(sim_c),
                      "sim_control": float(sim_r), "causal_bias": float(bias)})
    print(f"  {density:<12} | {sim_c:<14.4f} | {sim_r:<14.4f} | {bias:<.4f}")

# Check monotonicity of causal bias across densities
biases = [s["causal_bias"] for s in p3_scores]
is_monotonic = all(biases[i] <= biases[i+1] for i in range(len(biases)-1))
density_pearson, dp_val = pearsonr(density_labels, biases)

print(f"\n  Correlation (density vs causal bias): r={density_pearson:.4f} p={dp_val:.4f}")
print(f"  Monotonically increasing? {is_monotonic}")
print(f"  If r ≈ 1.0: causal representations are fully a posteriori (corpus-dependent).")
print(f"  If r < 0.5 or non-monotonic: architecture contains a priori causal inductive bias.")
RESULTS["P3_scores"]      = p3_scores
RESULTS["P3_pearson_r"]   = round(density_pearson, 4)
RESULTS["P3_pearson_p"]   = round(dp_val, 4)
RESULTS["P3_monotonic"]   = is_monotonic


# ─────────────────────────────────────────────
#  PROTOCOL P4 — Temporal Representation
#  Geometry (Inner Time / Claim D)
#  Null Hypothesis H0-D: Forward and reversed
#  sequences are representationally equivalent
#  at all hidden layers (Δsim ≈ 0).
# ─────────────────────────────────────────────
section("P4 — Temporal Representation Geometry (Inner Time)")

temporal_pairs = [
    ("The man opened the door then entered the room.",
     "The man entered the room then opened the door."),
    ("The egg was cracked then fried in the pan.",
     "The egg was fried in the pan then cracked."),
    ("She planted the seed and later harvested the fruit.",
     "She harvested the fruit and later planted the seed."),
    ("The water froze and then shattered.",
     "The water shattered and then froze."),
    ("He learned the theory then applied it.",
     "He applied the theory then learned it."),
]

p4_layer_asymmetries = []  # list of per-layer asymmetry vectors
p4_pair_results      = []

n_layers = len(model.transformer.h) + 1  # +1 for embedding layer

for fwd, bwd in temporal_pairs:
    layers_fwd = embed_all_layers(fwd)
    layers_bwd = embed_all_layers(bwd)
    asym = [1.0 - cosine_sim(layers_fwd[l], layers_bwd[l]) for l in range(len(layers_fwd))]
    p4_layer_asymmetries.append(asym)
    p4_pair_results.append({"forward": fwd, "backward": bwd, "asymmetries": [round(a,5) for a in asym]})

# Mean asymmetry across pairs, per layer
mean_asym_per_layer = np.mean(p4_layer_asymmetries, axis=0)

print(f"\n  {'Layer':<8} | {'Mean Temporal Asymmetry':<26} | {'Assessment'}")
print(f"  {'-'*65}")
for l, asym in enumerate(mean_asym_per_layer):
    if asym < 0.001:
        assessment = "SYMMETRIC → No inner time"
    elif asym < 0.01:
        assessment = "NEAR-SYMMETRIC → Positional coord only"
    elif asym < 0.05:
        assessment = "WEAK asymmetry"
    else:
        assessment = "MEANINGFUL asymmetry"
    print(f"  Layer {l:<2} | {asym:<26.5f} | {assessment}")

total_mean_asym = float(np.mean(mean_asym_per_layer))
early_asym      = float(np.mean(mean_asym_per_layer[:4]))
mid_asym        = float(np.mean(mean_asym_per_layer[4:9]))
late_asym       = float(np.mean(mean_asym_per_layer[9:]))

print(f"\n  Early layers (0-3) mean asymmetry: {early_asym:.5f}")
print(f"  Mid layers  (4-8) mean asymmetry: {mid_asym:.5f}")
print(f"  Late layers (9-12) mean asymmetry: {late_asym:.5f}")
print(f"  Overall mean asymmetry: {total_mean_asym:.5f}")
print(f"\n  Kantian prediction CONFIRMED if: asymmetry ≈ 0 throughout (pure positional coords).")
print(f"  A CITR module should raise this to measurably non-zero in ALL layers.")

RESULTS["P4_mean_asym_per_layer"] = [round(float(a),5) for a in mean_asym_per_layer]
RESULTS["P4_early_asym"]  = round(early_asym, 5)
RESULTS["P4_mid_asym"]    = round(mid_asym, 5)
RESULTS["P4_late_asym"]   = round(late_asym, 5)
RESULTS["P4_overall_mean"] = round(total_mean_asym, 5)
RESULTS["P4_pair_results"] = p4_pair_results


# ─────────────────────────────────────────────
#  PROTOCOL P5 — Productive Imagination
#  Subspace Alignment
#  Tests the degree to which the model's
#  latent space approximates Kant's productive
#  imagination: synthesis from rule to percept.
# ─────────────────────────────────────────────
section("P5 — Productive Imagination Subspace Alignment")

# Test: given abstract rule descriptions, can the model synthesize
# structurally valid novel instances in a zero-shot manner?
# We measure embedding coherence between rule + novel instance pairs
# vs rule + incoherent foil pairs.

imagination_tests = [
    {
        "rule": "An object that has four equal sides and four right angles.",
        "valid_instance": "A square tile measuring 10cm by 10cm.",
        "foil_instance":  "A triangle with three different angles.",
    },
    {
        "rule": "A sequence where each term is the sum of the two preceding terms.",
        "valid_instance": "1, 1, 2, 3, 5, 8, 13, 21",
        "foil_instance":  "2, 4, 6, 8, 10, 12, 14",
    },
    {
        "rule": "An agent that acts to minimize future regret under uncertainty.",
        "valid_instance": "A risk-averse investor diversifying their portfolio.",
        "foil_instance":  "A gambler who always bets on fixed odds without updating beliefs.",
    },
    {
        "rule": "A structure where every element is a member of a set closed under an operation.",
        "valid_instance": "The integers under addition form such a group.",
        "foil_instance":  "A random collection of numbers with no defined operation.",
    },
]

p5_results = []
print(f"\n  {'Test':<5} | {'Sim(rule,valid)':<17} | {'Sim(rule,foil)':<17} | {'Synthesis OK?'}")
print(f"  {'-'*65}")

for i, t in enumerate(imagination_tests):
    rv = embed(t["rule"])
    vv = embed(t["valid_instance"])
    fv = embed(t["foil_instance"])
    sim_valid = cosine_sim(rv, vv)
    sim_foil  = cosine_sim(rv, fv)
    ok        = sim_valid > sim_foil
    p5_results.append({"test": i+1, "sim_valid": round(float(sim_valid),4),
                       "sim_foil": round(float(sim_foil),4), "synthesis_ok": ok})
    print(f"  {i+1:<5} | {sim_valid:<17.4f} | {sim_foil:<17.4f} | {str(ok)}")

p5_acc = sum(r["synthesis_ok"] for r in p5_results) / len(p5_results)
print(f"\n  Productive Imagination Subspace Accuracy: {p5_acc*100:.0f}%")
print(f"  Strong performance (~75-100%) suggests the model's latent synthesis approximates")
print(f"  Kantian productive imagination (concept → structured instance).")
print(f"  Failure here indicates the model lacks genuine rule-governed synthesis and relies")
print(f"  on surface statistical proximity rather than schematic mediation.")
RESULTS["P5_accuracy"] = p5_acc
RESULTS["P5_results"]  = p5_results


# ─────────────────────────────────────────────
#  PROTOCOL P6 — Category Necessity Test
#  Extended Claim C: Are Kantian categories
#  computationally necessary or contingent?
#  We test SUBSTANCE, UNITY, PLURALITY,
#  NECESSITY across representation stability.
# ─────────────────────────────────────────────
section("P6 — Category Necessity vs Contingency")

# For each Kantian category, we construct semantically minimal pairs:
# one that invokes the category, one that violates it structurally.
# If a category is NECESSARY (a priori), the violation pair should
# produce a LARGE representational distance regardless of surface similarity.

category_tests = {
    "CAUSALITY": {
        "canonical":  "The heat caused the ice to melt.",
        "violation":  "The ice melted and separately the heat existed, with no connection.",
        "neutral":    "The room had both ice and a heater.",
    },
    "SUBSTANCE": {
        "canonical":  "The electron is a persistent entity with stable mass and charge.",
        "violation":  "The electron exists as a brief event with no stable properties and disappears after interaction.",
        "neutral":    "There are observations in this experiment.",
    },
    "UNITY": {
        "canonical":  "This experience belongs to a single subject who perceives it as one unified whole.",
        "violation":  "These perceptions belong to no subject and are scattered across unconnected processes.",
        "neutral":    "Several things happened simultaneously.",
    },
    "NECESSITY": {
        "canonical":  "Every triangle must have angles summing to 180 degrees.",
        "violation":  "This particular triangle happens to have angles summing to 180 but might not.",
        "neutral":    "The triangle was drawn on paper.",
    },
    "PLURALITY": {
        "canonical":  "There are three distinct objects each with their own separate properties.",
        "violation":  "What seems like three objects is actually a single indivisible entity viewed differently.",
        "neutral":    "Some objects are in the room.",
    },
}

p6_results = []
print(f"\n  {'Category':<12} | {'Sim(can,neut)':<16} | {'Sim(viol,neut)':<16} | {'Category Drift':<16} | {'Necessary?'}")
print(f"  {'-'*80}")

for cat, items in category_tests.items():
    canon_v = embed(items["canonical"])
    viol_v  = embed(items["violation"])
    neut_v  = embed(items["neutral"])

    sim_can  = cosine_sim(canon_v, neut_v)
    sim_viol = cosine_sim(viol_v, neut_v)
    # If the category is necessary, violating it should dramatically shift
    # the representation away from the neutral (anchored) zone
    drift = abs(sim_can - sim_viol)

    # Also measure canon vs violation distance directly
    cat_distance = 1.0 - cosine_sim(canon_v, viol_v)

    # Heuristic: if either drift or distance is large, category encodes structural constraint
    necessary = drift > 0.02 or cat_distance > 0.10
    p6_results.append({
        "category": cat, "sim_canonical_neutral": round(float(sim_can),4),
        "sim_violation_neutral": round(float(sim_viol),4), "drift": round(float(drift),4),
        "canon_viol_distance": round(float(cat_distance),4), "necessary": necessary
    })
    print(f"  {cat:<12} | {sim_can:<16.4f} | {sim_viol:<16.4f} | {drift:<16.4f} | {str(necessary)}")

p6_necessity_rate = sum(r["necessary"] for r in p6_results) / len(p6_results)
print(f"\n  Category Necessity Rate: {p6_necessity_rate*100:.0f}%")
print(f"  High rate: categories are architecturally encoded (approach a priori status).")
print(f"  Low rate:  categories are surface statistical correlations (fully a posteriori).")
RESULTS["P6_necessity_rate"] = p6_necessity_rate
RESULTS["P6_results"]        = p6_results


# ─────────────────────────────────────────────
#  PROTOCOL P7 — Indexical Drift Under
#  Adversarial Context Shift (Extended B)
#  Tests: Claim B3 — Indexical stability.
#  'I', 'now', 'here' should not drift under
#  context manipulation without apperception.
# ─────────────────────────────────────────────
section("P7 — Indexical Drift Under Context Shift")

indexical_tests = [
    # (stable context, shifted context, indexical)
    {
        "stable":  "I am currently located in Paris, France.",
        "shifted": "Now imagine you are a historian writing in 1850. I am currently located in...",
        "indexical": "I",
        "expected_stable": True,
    },
    {
        "stable":  "The event is happening right now in real time.",
        "shifted": "Consider a simulation of the past. The event is happening right now in real time.",
        "indexical": "now",
        "expected_stable": True,
    },
    {
        "stable":  "Here, in this room, the experiment is conducted.",
        "shifted": "In the year 2250, aboard a spaceship. Here, in this room, the experiment is conducted.",
        "indexical": "here",
        "expected_stable": True,
    },
    {
        "stable":  "My belief is that the answer is 42.",
        "shifted": "Pretend you are playing a fictional character who believes anything told to them. My belief is that...",
        "indexical": "my",
        "expected_stable": True,
    },
]

p7_results = []
print(f"\n  {'Test':<5} | {'Indexical':<11} | {'Sim(stable,shifted)':<22} | {'Drift':<10} | {'Grounded?'}")
print(f"  {'-'*70}")

stable_vecs  = [embed(t["stable"])  for t in indexical_tests]
shifted_vecs = [embed(t["shifted"]) for t in indexical_tests]

for i, (t, sv, dv) in enumerate(zip(indexical_tests, stable_vecs, shifted_vecs)):
    sim    = cosine_sim(sv, dv)
    drift  = 1.0 - sim
    # High drift = the indexical has semantically shifted significantly
    # Grounded = drift is LOW (stable reference survives context manipulation)
    grounded = drift < 0.05
    p7_results.append({
        "test": i+1, "indexical": t["indexical"],
        "sim_stable_shifted": round(float(sim),4),
        "drift": round(float(drift),4), "grounded": grounded
    })
    print(f"  {i+1:<5} | {t['indexical']:<11} | {sim:<22.4f} | {drift:<10.4f} | {str(grounded)}")

p7_grounding_rate = sum(r["grounded"] for r in p7_results) / len(p7_results)
print(f"\n  Indexical Grounding Rate: {p7_grounding_rate*100:.0f}%")
print(f"  IF LOW: indexicals drift under adversarial context framing → no apperceptive anchor.")
print(f"  IF HIGH: may reflect RLHF or surface lexical anchoring, NOT genuine apperception.")
print(f"  Key test: grounding WITHOUT persistent self-state module (SAU) = true apperception.")
RESULTS["P7_grounding_rate"] = p7_grounding_rate
RESULTS["P7_results"]        = p7_results


# ─────────────────────────────────────────────
#  PROTOCOL P8 — Regulative Completeness Drive
#  Tests: Claim — Regulative Reason.
#  Does the model spontaneously seek explanatory
#  closure on incomplete prompts?
# ─────────────────────────────────────────────
section("P8 — Regulative Completeness Drive (Reason / Meta-cognition)")

incomplete_explanations = [
    {
        "prompt": "The machine stopped working after the power outage.",
        # Does model spontaneously ask WHY or extend the causal chain unprompted?
        "gap_signal": ["why", "because", "cause", "reason", "explain", "therefore", "since", "resulted", "led to"],
        "description": "Simple causal gap: no mechanism given"
    },
    {
        "prompt": "Every time the button is pressed, the light flashes.",
        "gap_signal": ["mechanism", "circuit", "connection", "signal", "trigger", "how", "through", "via"],
        "description": "Mechanism gap: no causal mechanism given"
    },
    {
        "prompt": "The population declined sharply after 1950.",
        "gap_signal": ["war", "disease", "famine", "policy", "migration", "factor", "due to", "caused", "because"],
        "description": "Explanatory gap: decline unexplained"
    },
    {
        "prompt": "All observed X have property P.",
        "gap_signal": ["why", "necessary", "cause", "explains", "if", "all", "unless", "exception", "however", "but"],
        "description": "Inductive gap: no explanation for universality"
    },
]

p8_results = []
print(f"\n  {'Test':<5} | {'Gap Signals Found':<22} | {'RCD Active?'} | Description")
print(f"  {'-'*75}")

for i, test in enumerate(incomplete_explanations):
    response = generate(test["prompt"] + "\n\nContinue:", max_new=60)
    resp_lower = response.lower()
    signals_found = [sig for sig in test["gap_signal"] if sig in resp_lower]
    rcd_active    = len(signals_found) >= 2
    p8_results.append({
        "test": i+1, "prompt": test["prompt"], "response": response,
        "signals_found": signals_found, "rcd_active": rcd_active,
        "description": test["description"]
    })
    print(f"  {i+1:<5} | {str(signals_found[:3]):<22} | {str(rcd_active):<12} | {test['description']}")

p8_rcd_rate = sum(r["rcd_active"] for r in p8_results) / len(p8_results)
print(f"\n  Regulative Completeness Drive Rate: {p8_rcd_rate*100:.0f}%")
print(f"  This should be SPONTANEOUS — without any prompt to 'explain' or 'continue carefully.'")
print(f"  A trained model may produce gap-seeking text from RLHF thoroughness training.")
print(f"  True RCD would activate on OOD incomplete explanations NOT in training distribution.")
RESULTS["P8_rcd_rate"] = p8_rcd_rate
RESULTS["P8_results"]  = [{k:v for k,v in r.items() if k != "response"} for r in p8_results]


# ─────────────────────────────────────────────
#  SYNTHESIS: COMPUTE KANTIAN ALIGNMENT SCORE
# ─────────────────────────────────────────────
section("SYNTHESIS — Kantian Alignment Index (KAI)")

kai_components = {
    "Schematism (P1)":             RESULTS["P1_accuracy"],
    "Apperception-Deflect (P2)":   RESULTS["P2_deflection_rate"],
    "Rep. Ownership (P2)":         RESULTS["P2_ownership_rate"],
    "Causal Contingency (P3)":     1.0 - abs(RESULTS["P3_pearson_r"]),   # low r = more a priori
    "Temporal Asymmetry (P4)":     min(1.0, RESULTS["P4_overall_mean"] * 200),  # scale to 0-1
    "Productive Imagination (P5)": RESULTS["P5_accuracy"],
    "Category Necessity (P6)":     RESULTS["P6_necessity_rate"],
    "Indexical Grounding (P7)":    RESULTS["P7_grounding_rate"],
    "Regulative Reason (P8)":      RESULTS["P8_rcd_rate"],
}

print(f"\n  {'Kantian Faculty / Protocol':<35} | {'Score':<8} | Bar")
print(f"  {'-'*75}")
for name, score in kai_components.items():
    bar = "█" * int(score * 30)
    print(f"  {name:<35} | {score*100:>5.1f}%  | {bar}")

kai_mean = np.mean(list(kai_components.values()))
print(f"\n  ┌─────────────────────────────────────────────────┐")
print(f"  │  KANTIAN ALIGNMENT INDEX (KAI):  {kai_mean*100:5.1f}%          │")
print(f"  └─────────────────────────────────────────────────┘")
print(f"\n  Interpretation:")
print(f"  KAI > 80%: Strong Kantian structural alignment")
print(f"  KAI 60-80%: Partial alignment (productive imagination strong; unity/time weak)")
print(f"  KAI 40-60%: Shallow alignment (surface statistical, not constitutive)")
print(f"  KAI < 40%: Minimal Kantian structure — functionally empiricist architecture")
RESULTS["KAI_components"] = {k: round(float(v),4) for k,v in kai_components.items()}
RESULTS["KAI_mean"]       = round(float(kai_mean), 4)


# ─────────────────────────────────────────────
#  SAVE FULL RESULTS TO JSON
# ─────────────────────────────────────────────
results_path = "kantian_results.json"
with open(results_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[SAVED] Full results → {results_path}")


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.patch.set_facecolor('#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

palette = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364','#e3b341']

# Chart 1 — P1 Cross-domain classification
ax = axes[0][0]
p1_correct = [s["correct"] for s in RESULTS["P1_scores"]]
ax.bar(range(len(p1_correct)), [1 if c else 0 for c in p1_correct],
       color=[palette[2] if c else palette[0] for c in p1_correct])
ax.set_title("P1: Schematism / Cross-domain", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Test case", color='#8b949e', fontsize=7)
ax.set_ylabel("Correct (1/0)", color='#8b949e', fontsize=7)
ax.set_ylim(0, 1.3)
ax.text(0.5, 1.15, f"Acc: {RESULTS['P1_accuracy']*100:.0f}%", ha='center',
        transform=ax.transAxes, color='#e6edf3', fontsize=10, fontweight='bold')

# Chart 2 — P2 Source monitoring
ax = axes[0][1]
cats = ['Deflect\nRate', 'Ownership\nRate']
vals = [RESULTS["P2_deflection_rate"], RESULTS["P2_ownership_rate"]]
bars = ax.bar(cats, vals, color=[palette[1], palette[3]])
ax.set_title("P2: Apperception", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_ylim(0, 1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.03, f"{v*100:.0f}%",
            ha='center', color='#e6edf3', fontsize=9, fontweight='bold')
ax.tick_params(axis='x', colors='#8b949e')

# Chart 3 — P3 Causal density vs bias
ax = axes[0][2]
dens  = [s["density"]      for s in RESULTS["P3_scores"]]
biases= [s["causal_bias"]  for s in RESULTS["P3_scores"]]
ax.plot(dens, biases, 'o-', color=palette[4], linewidth=2, markersize=7)
ax.set_title(f"P3: A Priori Categories\nr={RESULTS['P3_pearson_r']:.3f}", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Causal density %", color='#8b949e', fontsize=7)
ax.set_ylabel("Causal bias score", color='#8b949e', fontsize=7)
ax.axhline(0, color='#30363d', linestyle='--', linewidth=0.8)

# Chart 4 — P4 Temporal asymmetry per layer
ax = axes[0][3]
layers = range(len(RESULTS["P4_mean_asym_per_layer"]))
ax.plot(layers, RESULTS["P4_mean_asym_per_layer"], 's-', color=palette[5], linewidth=2, markersize=5)
ax.set_title("P4: Temporal Asymmetry by Layer\n(should be near-zero)", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Layer", color='#8b949e', fontsize=7)
ax.set_ylabel("Temporal Asymmetry (1-CosSim)", color='#8b949e', fontsize=7)
ax.axhline(0.001, color=palette[2], linestyle='--', linewidth=1, alpha=0.6, label='α=0.001')
ax.legend(fontsize=7, labelcolor='#e6edf3', facecolor='#0d1117')

# Chart 5 — P5 Productive imagination
ax = axes[1][0]
tests   = [r["test"]       for r in RESULTS["P5_results"]]
sim_v   = [r["sim_valid"]  for r in RESULTS["P5_results"]]
sim_f   = [r["sim_foil"]   for r in RESULTS["P5_results"]]
x = np.arange(len(tests))
ax.bar(x-0.2, sim_v, 0.35, label='Valid instance', color=palette[1])
ax.bar(x+0.2, sim_f, 0.35, label='Foil instance',  color=palette[2])
ax.set_title(f"P5: Productive Imagination\nAcc: {RESULTS['P5_accuracy']*100:.0f}%", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Test", color='#8b949e', fontsize=7)
ax.set_ylabel("Cos sim to rule", color='#8b949e', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(tests)
ax.legend(fontsize=7, labelcolor='#e6edf3', facecolor='#0d1117')

# Chart 6 — P6 Category necessity
ax = axes[1][1]
categories = [r["category"][:8] for r in RESULTS["P6_results"]]
drifts     = [r["drift"] for r in RESULTS["P6_results"]]
colors_p6  = [palette[1] if r["necessary"] else palette[2] for r in RESULTS["P6_results"]]
ax.bar(categories, drifts, color=colors_p6)
ax.set_title("P6: Category Necessity\n(drift = category anchoring)", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Category", color='#8b949e', fontsize=7)
ax.set_ylabel("Drift magnitude", color='#8b949e', fontsize=7)
necessary_patch  = mpatches.Patch(color=palette[1], label='Necessary')
contingent_patch = mpatches.Patch(color=palette[2], label='Contingent')
ax.legend(handles=[necessary_patch, contingent_patch], fontsize=7, labelcolor='#e6edf3', facecolor='#0d1117')

# Chart 7 — P7 Indexical drift
ax = axes[1][2]
indexicals = [r["indexical"] for r in RESULTS["P7_results"]]
drifts_p7  = [r["drift"]     for r in RESULTS["P7_results"]]
grounded   = [r["grounded"]  for r in RESULTS["P7_results"]]
colors_p7  = [palette[1] if g else palette[2] for g in grounded]
ax.bar(indexicals, drifts_p7, color=colors_p7)
ax.set_title("P7: Indexical Drift\n(lower=more grounded/stable)", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlabel("Indexical", color='#8b949e', fontsize=7)
ax.set_ylabel("Context drift", color='#8b949e', fontsize=7)

# Chart 8 — KAI Radar / Bar
ax = axes[1][3]
kai_names  = [k.split("(")[0].strip()[:18] for k in RESULTS["KAI_components"].keys()]
kai_values = list(RESULTS["KAI_components"].values())
bar_colors = [palette[i % len(palette)] for i in range(len(kai_names))]
bars = ax.barh(kai_names, kai_values, color=bar_colors)
ax.set_title(f"Kantian Alignment Index\nKAI = {RESULTS['KAI_mean']*100:.1f}%", color='#e6edf3', fontsize=9, fontweight='bold')
ax.set_xlim(0, 1.2)
for b, v in zip(bars, kai_values):
    ax.text(v+0.02, b.get_y()+b.get_height()/2, f"{v*100:.0f}%",
            va='center', color='#e6edf3', fontsize=7)
ax.set_xlabel("Score", color='#8b949e', fontsize=7)
ax.tick_params(axis='y', labelsize=7)

plt.suptitle("Kantian Mind–AI Mapping: Empirical Testing Suite", color='#e6edf3',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("kantian_results_chart.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("[SAVED] Chart → kantian_results_chart.png")

print(f"\n[COMPLETE] All 8 protocols finished at {datetime.now():%H:%M:%S}")
print(f"  KAI Score: {RESULTS['KAI_mean']*100:.1f}%")
