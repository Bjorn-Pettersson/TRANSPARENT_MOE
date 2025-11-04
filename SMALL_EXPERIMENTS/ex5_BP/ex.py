# transformer_moe_allpretrain_experiment_topk_seq.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
from transformers import GPT2Tokenizer
import seaborn as sns
import csv

# -------------------- Config --------------------
CONFIG = {
    "hidden_size": 256,
    "num_experts": 4,
    "expert_layers": 1,
    "pretrain_epochs": 2,        # light pretraining
    "pretrain_n_per_domain": 300,
    "moe_train_epochs": 6,
    "moe_n_per_domain": 600,
    "batch_size": 16,
    "lr": 1e-4,
    "seed_list": [42],
    "use_pretrained_embeddings": False,
    "freeze_pretrained_for": 0,
    "entropy_reg": 0.0,
    "top_k": 2,                  # top-k for sequence-level routing
    "save_dir": "moe_allpretrain_results_topk_seq"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# -------------------- Dataset (math, law, biokem, stroy) --------------------
class SpecializedTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, dataset_type='mixed', n_per_domain=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        n = n_per_domain

        domains = {
            'math': self._generate_math_samples(n),
            'law': self._generate_law_samples(n),
            'biokem': self._generate_biokem_samples(n),
            'stroy': self._generate_stroy_samples(n)
        }

        if dataset_type in domains:
            self.data = domains[dataset_type]
            self.labels = [dataset_type] * len(self.data)
        elif dataset_type == 'mixed':
            for domain, samples in domains.items():
                self.data.extend(samples)
                self.labels.extend([domain] * len(samples))
        else:
            raise ValueError("dataset_type must be one of the domains or 'mixed'")

        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        if combined:
            self.data, self.labels = zip(*combined)
        else:
            self.data, self.labels = [], []

    # --- generators ---
    def _generate_math_samples(self, n):
        templates = [
            "Consider the function f(x) = {poly}. Compute f'({x0}).",
            "Prove that the sequence a_n = {recurrence} converges and find its limit.",
            "Let A be a matrix with eigenvalues {eig1} and {eig2}; determine if A is diagonalizable.",
            "Evaluate the integral ∫ {expr} dx between {a} and {b}.",
            "Solve the differential equation y' + {p} y = {q} with initial condition y({x0}) = {y0}."
        ]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                poly=f"{random.randint(1,5)}x^{random.randint(1,4)} + {random.randint(0,5)}x + {random.randint(0,3)}",
                x0=random.randint(0,5),
                recurrence=f"a_n = a_{{n-1}}/{random.randint(2,5)} + {random.randint(0,3)}",
                eig1=random.randint(-3,3),
                eig2=random.randint(-3,3),
                expr=f"{random.randint(1,5)}*x^{random.randint(0,3)}",
                a=random.randint(-2,0),
                b=random.randint(1,4),
                p=random.randint(0,5),
                q=random.randint(0,5),
                y0=random.randint(0,3)
            )
            samples.append(s)
        return samples

    def _generate_law_samples(self, n):
        templates = [
            "Under the {act}, courts have held that {principle} applies when {condition}.",
            "The contract was voidable due to {defect}; relevant precedent includes {case}.",
            "A plaintiff must establish {element1}, {element2}, and {element3} to succeed in a negligence claim.",
            "Statutory interpretation favored the narrow reading because the legislative history showed {reason}."
        ]
        acts = ["Contracts Act", "Tort Law Reform", "Evidence Code"]
        principles = ["strict liability", "reasonable person standard", "proportionality"]
        cases = ["Case A v. B", "R v. Smith", "Brown v. Board"]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                act=random.choice(acts),
                principle=random.choice(principles),
                condition=random.choice(["harm was foreseeable", "parties had unequal bargaining power"]),
                defect=random.choice(["misrepresentation", "undue influence", "illegality"]),
                case=random.choice(cases),
                element1="duty",
                element2="breach",
                element3="causation",
                reason=random.choice(["clear legislative intent", "policy concerns"])
            )
            samples.append(s)
        return samples

    def _generate_biokem_samples(self, n):
        # 'biokem' = biochemical / biochemical kinetics style text
        templates = [
            "The enzyme follows Michaelis-Menten kinetics with Vmax = {vmax} and Km = {km}.",
            "Spectroscopy revealed absorbance peaks at {nm} nm indicating the presence of {compound}.",
            "The reaction rate increased with substrate concentration until it reached Vmax, suggesting {interpretation}.",
            "Cell cultures treated with {agent} showed upregulated expression of {gene}."
        ]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                vmax=round(random.uniform(0.5, 10.0), 2),
                km=round(random.uniform(0.01, 5.0), 3),
                nm=random.randint(200, 700),
                compound=random.choice(["heme", "chlorophyll", "aromatic amino acid"]),
                interpretation=random.choice(["enzyme saturation", "allosteric regulation"]),
                agent=random.choice(["DrugX", "inhibitorY"]),
                gene=random.choice(["GeneA", "ProteinB"])
            )
            samples.append(s)
        return samples

    def _generate_stroy_samples(self, n):
        # 'stroy' is treated as short creative story-like lines
        templates = [
            "He found the letter on the kitchen table and the edges were {adj}.",
            "Rain came down in {pattern} as she walked toward the {place}.",
            "The city at night smelled of {smell} and old concrete, and the neon was {color}.",
            "She remembered her childhood by the sea: the {object}, the {sound}, and the blue horizon."
        ]
        adjectives = ["torn", "crisp", "watermarked"]
        patterns = ["sheets", "a steady drizzle", "a sudden downpour"]
        places = ["station", "apartment", "pier"]
        smells = ["salt", "fried bread", "wet asphalt"]
        colors = ["flickering", "blinding", "muted"]
        objects = ["guitar", "toy boat", "old tin"]
        sounds = ["distant laughter", "a train's whistle", "seagulls"]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                adj=random.choice(adjectives),
                pattern=random.choice(patterns),
                place=random.choice(places),
                smell=random.choice(smells),
                color=random.choice(colors),
                object=random.choice(objects),
                sound=random.choice(sounds)
            )
            samples.append(s)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'domain': label
        }

# -------------------- Model --------------------
class TransformerExpert(nn.Module):
    def __init__(self, hidden_size=CONFIG["hidden_size"], num_layers=CONFIG["expert_layers"]):
        super().__init__()
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        out = self.transformer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        return out

class RouterSequenceTopK(nn.Module):
    """
    Sequence-level router that computes gating per sequence (per example),
    selects top-k experts per sequence and returns sparse normalized weights.
    Outputs:
      - seq_weights: [B, E] (sparse; zeros for non-topk)
      - expanded_weights: [B, L, E] (same weights expanded across sequence length)
    """
    def __init__(self, hidden_size=CONFIG["hidden_size"], num_experts=CONFIG["num_experts"], top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_experts)
        )

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H]
        # pool to sequence vector (mean pooling over valid tokens)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)  # [B,1]
            summed = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)  # [B,H]
            pooled = summed / (lengths.clamp(min=1).to(hidden_states.dtype))
        else:
            pooled = hidden_states.mean(dim=1)  # [B,H]

        logits = self.gate(pooled)  # [B, E]

        # select top-k indices per sequence
        topk_vals, topk_idx = torch.topk(logits, k=min(self.top_k, self.num_experts), dim=-1)  # [B, k]
        # Build sparse mask and compute softmax over just the topk logits
        B = logits.size(0)
        device = logits.device
        sparse_logits = torch.full_like(logits, float('-inf'))  # [B, E]
        arange = torch.arange(B, device=device).unsqueeze(-1)
        sparse_logits[arange, topk_idx] = topk_vals
        # now safe softmax (will be zero where -inf)
        seq_weights = F.softmax(sparse_logits, dim=-1)  # [B, E], zeros outside topk
        # expand across sequence length
        L = hidden_states.size(1)
        expanded = seq_weights.unsqueeze(1).expand(-1, L, -1)  # [B, L, E]
        return seq_weights, expanded  # sequence weights and expanded token-level weights

class TransformerMoE(nn.Module):
    def __init__(self, vocab_size, hidden_size=CONFIG["hidden_size"], num_experts=CONFIG["num_experts"], num_layers=CONFIG["expert_layers"], top_k=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.router = RouterSequenceTopK(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([TransformerExpert(hidden_size, num_layers) for _ in range(num_experts)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask=None, return_routing=False):
        # input_ids: [B, L]
        hidden = self.embeddings(input_ids)  # [B, L, H]
        seq_weights, routing_expanded = self.router(hidden, attention_mask)  # [B, E], [B, L, E]

        # run each expert on the same hidden inputs
        expert_outputs = []
        for e in range(self.num_experts):
            out = self.experts[e](hidden, attention_mask)  # [B, L, H]
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [E, B, L, H]

        # prepare routing for combination: want [E, B, L, 1]
        # routing_expanded: [B, L, E] -> permute to [E, B, L]
        r = routing_expanded.permute(2, 0, 1).unsqueeze(-1)  # [E, B, L, 1]
        mixed = (expert_outputs * r).sum(dim=0)  # [B, L, H]

        logits = self.lm_head(mixed)  # [B, L, V]
        if return_routing:
            # return both sequence-level sparse weights and expanded token-level weights
            return logits, seq_weights, routing_expanded
        return logits

    def load_pretrained_expert(self, expert_idx, state_dict):
        self.experts[expert_idx].load_state_dict(state_dict)
        print(f"Loaded pretrained expert into slot {expert_idx}")

# -------------------- Pretrain each expert lightly --------------------
def pretrain_all_experts_light(tokenizer, vocab_size, device, seed=42):
    """
    Pretrain each expert lightly on its own domain and return lists of expert state_dicts and embedding state_dicts.
    """
    print("\n== PHASE 1: Lightly pretrain ALL experts (per-domain) ==")
    domain_order = ['math', 'law', 'biokem', 'stroy']
    pre_states = []
    pre_embeddings = []

    for i, domain in enumerate(domain_order):
        print(f"\nPretraining expert {i} on domain '{domain}' (light)...")
        # generate domain-specific samples
        tmp = SpecializedTextDataset(tokenizer, dataset_type='mixed', n_per_domain=0)
        if domain == 'math':
            samples = tmp._generate_math_samples(CONFIG["pretrain_n_per_domain"])
        elif domain == 'law':
            samples = tmp._generate_law_samples(CONFIG["pretrain_n_per_domain"])
        elif domain == 'biokem':
            samples = tmp._generate_biokem_samples(CONFIG["pretrain_n_per_domain"])
        elif domain == 'stroy':
            samples = tmp._generate_stroy_samples(CONFIG["pretrain_n_per_domain"])
        else:
            raise ValueError("Unknown domain")

        labels = [domain] * len(samples)

        class SmallTextDS(Dataset):
            def __init__(self, samples, labels, tokenizer):
                self.samples = samples
                self.labels = labels
                self.tokenizer = tokenizer
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                encoded = self.tokenizer(self.samples[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
                return {'input_ids': encoded['input_ids'].squeeze(0), 'attention_mask': encoded['attention_mask'].squeeze(0), 'domain': self.labels[idx]}

        dataset = SmallTextDS(samples, labels, tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

        expert = TransformerExpert(CONFIG["hidden_size"], CONFIG["expert_layers"]).to(device)
        emb = nn.Embedding(vocab_size, CONFIG["hidden_size"]).to(device)
        lm_head = nn.Linear(CONFIG["hidden_size"], vocab_size).to(device)

        opt = optim.Adam(list(expert.parameters()) + list(emb.parameters()) + list(lm_head.parameters()), lr=CONFIG["lr"])
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for epoch in range(CONFIG["pretrain_epochs"]):
            expert.train()
            total = 0.0
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hidden = emb(input_ids)
                out = expert(hidden, attention_mask)
                logits = lm_head(out)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  Epoch {epoch+1}/{CONFIG['pretrain_epochs']} avg loss: {total/len(loader):.4f}")

        pre_states.append(expert.state_dict())
        pre_embeddings.append(emb.state_dict())

    print("\nAll experts lightly pretrained.")
    return pre_states, pre_embeddings

# -------------------- Train MoE --------------------
def train_moe_allpretrained(pre_states, pre_embeddings, tokenizer, vocab_size, device, seed=42):
    print("\n== PHASE 2: Initialize MoE with pretrained experts and train on mixed data (top-k seq routing) ==")
    mixed = SpecializedTextDataset(tokenizer, dataset_type='mixed', n_per_domain=CONFIG["moe_n_per_domain"])
    loader = DataLoader(mixed, batch_size=CONFIG["batch_size"], shuffle=True)

    moe = TransformerMoE(vocab_size, CONFIG["hidden_size"], CONFIG["num_experts"], CONFIG["expert_layers"], top_k=CONFIG["top_k"]).to(device)

    # Load expert states into slots
    for i, st in enumerate(pre_states):
        moe.load_pretrained_expert(i, st)

    if CONFIG["use_pretrained_embeddings"]:
        avg_weight = None
        for d in pre_embeddings:
            w = d['weight']
            if avg_weight is None:
                avg_weight = w.clone()
            else:
                avg_weight += w
        avg_weight /= len(pre_embeddings)
        moe.embeddings.weight.data.copy_(avg_weight)
        print("Loaded averaged pretrained embeddings into MoE embeddings.")
    else:
        print("MoE embeddings left random (use_pretrained_embeddings=False).")

    opt = optim.Adam(moe.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    freeze_k = CONFIG["freeze_pretrained_for"]

    for epoch in range(CONFIG["moe_train_epochs"]):
        if epoch < freeze_k:
            for p in moe.experts:
                for param in p.parameters():
                    param.requires_grad = False
            print(f"Epoch {epoch+1}: pretrained experts frozen.")
        elif epoch == freeze_k:
            for p in moe.experts:
                for param in p.parameters():
                    param.requires_grad = True
            print(f"Epoch {epoch+1}: unfroze pretrained experts; full fine-tuning now.")

        moe.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits, seq_routing, expanded_routing = moe(input_ids, attention_mask, return_routing=True)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # entropy regularizer on the sequence-level routing distribution (encourage spread)
            if CONFIG["entropy_reg"] > 0.0:
                # seq_routing: [B, E]
                entropy = - (seq_routing * (seq_routing + 1e-12).log()).sum(dim=-1).mean()
                loss = loss + CONFIG["entropy_reg"] * (-entropy)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{CONFIG['moe_train_epochs']} avg loss: {total_loss/len(loader):.4f}")

    print("MoE training finished.")
    return moe

# -------------------- Analysis --------------------
def analyze_routing(moe, tokenizer, device):
    print("\n== PHASE 3: Analyze sequence-level top-k routing (domain <-> expert alignment) ==")
    test_samples = {
        'math': [
            "Compute the derivative of f(x) = 3x^3 + 2x - 5 at x = 2.",
            "Find the limit of the sequence defined by a_n = a_{n-1}/2 + 1 with a_0 = 1.",
            "Evaluate the integral of x^2 from 0 to 3.",
            "Is the matrix with eigenvalues 1 and 2 diagonalizable?",
            "Solve y' + 2y = 3 with initial condition y(0)=0."
        ],
        'law': [
            "Under the Contracts Act, misrepresentation can make a contract voidable.",
            "A negligence claim requires duty, breach, and causation to be established.",
            "The court relied on precedent in R v. Smith to interpret the statute.",
            "Public policy concerns influenced the judgment on enforceability.",
            "The defendant argued undue influence rendered the agreement void."
        ],
        'biokem': [
            "The enzyme kinetics gave Vmax = 5.2 and Km = 0.15 consistent with saturation.",
            "Absorbance peaked at 280 nm suggesting aromatic residues present.",
            "Substrate concentration increase led to rate saturation at high levels.",
            "Cells treated with inhibitorY showed decreased expression of ProteinB.",
            "Michaelis-Menten parameters were estimated from Lineweaver-Burk plots."
        ],
        'stroy': [
            "She opened the letter and the edges were crisp like old paper.",
            "Rain fell in sheets as he ran toward the pier.",
            "The city smelled of salt and neon flickered above the wet streets.",
            "He remembered a toy boat and the sound of seagulls from childhood.",
            "The kitchen table held a cup and a folded map with faded ink."
        ]
    }

    domain_order = ['math', 'law', 'biokem', 'stroy']
    routing_stats = defaultdict(list)
    seq_topk_counts = defaultdict(Counter)  # count which experts are in top-k for each domain

    moe.eval()
    with torch.no_grad():
        for domain, samples in test_samples.items():
            for text in samples:
                enc = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                logits, seq_routing, expanded_routing = moe(input_ids, attention_mask, return_routing=True)
                # seq_routing: [B, E], batch size 1
                seq_r = seq_routing[0].cpu().numpy()
                routing_stats[domain].append(seq_r)
                # which are top-k (non-zero in seq_r typically)
                topk_indices = np.flatnonzero(seq_r > 0)
                for idx in topk_indices:
                    seq_topk_counts[domain][int(idx)] += 1

    # Build usage matrix (average % weight per expert across sequences)
    usage = np.zeros((len(domain_order), moe.num_experts))
    for i, d in enumerate(domain_order):
        arr = np.array(routing_stats[d]) if routing_stats[d] else np.zeros((1, moe.num_experts))
        usage[i] = arr.mean(axis=0) * 100

    # Print per-domain expert breakdown
    print("\nAverage expert usage (%) by domain (sequence-level):")
    for i, d in enumerate(domain_order):
        print(f"\n{d.upper()}:")
        for e in range(moe.num_experts):
            print(f"  Expert {e}: {usage[i, e]:.2f}%")

    # Compute "matching expert" statistics:
    matches = []
    for i, d in enumerate(domain_order):
        counts = seq_topk_counts[d]
        total = sum(counts.values()) if counts else 0
        # count how often canonical expert i appears in top-k for domain d
        match_count = counts.get(i, 0)
        frac = match_count / total if total > 0 else 0.0
        matches.append((d, i, match_count, total, frac))
        print(f"\nDomain '{d}': expert {i} present in top-k for {match_count}/{total} sequences = {frac*100:.1f}%")

    # Save CSV summary
    csv_path = os.path.join(CONFIG["save_dir"], f"routing_summary_seed_seqtopk.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "matching_expert", "match_count", "total_sequences", "fraction"])
        for d, idx, mcount, tot, frac in matches:
            writer.writerow([d, idx, mcount, tot, frac])

    # Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(usage, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=[f'E{e}' for e in range(moe.num_experts)],
                yticklabels=[d.capitalize() for d in domain_order])
    plt.title('Average Expert Usage (%) by Domain (sequence-level)')
    plt.xlabel('Expert')
    plt.ylabel('Domain')
    plt.tight_layout()
    imgpath = os.path.join(CONFIG["save_dir"], "usage_heatmap_seqtopk.png")
    plt.savefig(imgpath, dpi=150, bbox_inches='tight')
    plt.close()

    return usage, seq_topk_counts, matches

# -------------------- Multi-seed runner --------------------
def run_single_seed(seed):
    print(f"\n\n=== Running seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    pre_states, pre_embeds = pretrain_all_experts_light(tokenizer, vocab_size, device, seed)
    moe = train_moe_allpretrained(pre_states, pre_embeds, tokenizer, vocab_size, device, seed)
    usage, seq_topk_counts, matches = analyze_routing(moe, tokenizer, device)

    fractions = {m[0]: m[4] for m in matches}
    return fractions, usage

def run_multi_seed():
    all_results = []
    usages = []
    for seed in CONFIG["seed_list"]:
        fractions, usage = run_single_seed(seed)
        all_results.append(fractions)
        usages.append(usage)
    # Aggregate
    domains = ['math', 'law', 'biokem', 'stroy']
    summary = {}
    for d in domains:
        vals = [res[d] for res in all_results]
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        summary[d] = {"mean": mean, "std": std, "values": vals}
        print(f"\nDomain {d}: matched-expert fraction mean={mean*100:.1f}%, std={std*100:.1f}% over {len(vals)} seeds")

    # Save CSV
    csv_out = os.path.join(CONFIG["save_dir"], "multi_seed_summary_seqtopk.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "mean_frac", "std_frac"] + [f"seed_{s}" for s in CONFIG["seed_list"]])
        for d in domains:
            row = [d, summary[d]["mean"], summary[d]["std"]] + summary[d]["values"]
            writer.writerow(row)

    # Save aggregated usage heatmap (mean across seeds)
    mean_usage = np.mean(np.stack(usages, axis=0), axis=0)  # shape: seeds x domains x experts -> mean across seeds
    plt.figure(figsize=(8,6))
    sns.heatmap(mean_usage, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[f'E{e}' for e in range(CONFIG["num_experts"])],
                yticklabels=[d.capitalize() for d in domains])
    plt.title('Mean Expert Usage (%) by Domain (across seeds) - seq top-k')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "mean_usage_heatmap_seqtopk.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return summary

# -------------------- Entry --------------------
if __name__ == "__main__":
    summary = run_multi_seed()
    print("\nAll experiments done. Results saved to:", CONFIG["save_dir"])
    print("Key outputs: CSV 'multi_seed_summary_seqtopk.csv' and heatmap images.")
