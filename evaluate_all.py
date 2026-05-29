import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import os

from Aster import TransformerGPT, load_model_best_state_dict

class AsterOBQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = []
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        with open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                tokens     = line.split('|')
                fact       = tokens[0]
                stem       = tokens[1]
                choices    = [tokens[2], tokens[3], tokens[4], tokens[5]]
                answer_key = tokens[6]
                label_idx  = self.label_map[answer_key]

                prompt_text    = f"{fact} {stem}"
                prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)['input_ids']

                all_input_ids      = []
                all_attention_mask = []
                all_labels         = []

                for choice_text in choices:
                    choice_encoded = tokenizer(
                        " " + choice_text + (tokenizer.eos_token or ""),
                        add_special_tokens=False
                    )['input_ids']

                    input_ids      = prompt_encoded + choice_encoded
                    labels         = [-100] * len(prompt_encoded) + choice_encoded.copy()
                    attention_mask = [1] * len(input_ids)

                    pad_len = max_length - len(input_ids)
                    if pad_len > 0:
                        input_ids.extend([tokenizer.pad_token_id] * pad_len)
                        labels.extend([-100] * pad_len)
                        attention_mask.extend([0] * pad_len)
                    else:
                        input_ids      = input_ids[:max_length]
                        labels         = labels[:max_length]
                        attention_mask = attention_mask[:max_length]

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)

                self.data.append({
                    "input_ids":      torch.tensor(all_input_ids),
                    "attention_mask": torch.tensor(all_attention_mask),
                    "labels":         torch.tensor(all_labels),
                    "label_idx":      label_idx,
                    "raw_fact":       fact,
                    "raw_stem":       stem,
                    "raw_choices":    choices,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AsterPipeline:
    def __init__(self, tokenizer_dir="saved/tokenizer", model_path="saved/model_best.pt"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        state_dict = load_model_best_state_dict(model_path)
        vocab_size  = int(state_dict["wte.weight"].shape[0])

        self.model = TransformerGPT(vocab_size=vocab_size, d_model=1024, n_layers=16,
                                    heads=16, seqlen=1024, d_ff=4096)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def sequence_probability(self, prompt, choice):
        self.model.eval()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        choice_ids = self.tokenizer(" " + choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        input_ids = torch.cat([prompt_ids, choice_ids], dim=1)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            _, logits  = self.model(input_ids)
            log_probs  = F.log_softmax(logits, dim=-1)

            seq_log_prob  = 0.0
            prompt_len    = prompt_ids.size(1)
            n_choice_toks = choice_ids.size(1)

            for i in range(prompt_len - 1, prompt_len + n_choice_toks - 1):
                target_token  = input_ids[0, i + 1]
                seq_log_prob += log_probs[0, i, target_token].item()

        return seq_log_prob / (n_choice_toks ** 0.7)

    def compute_bertscore(self, text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        ids1 = self.tokenizer(text1, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        ids2 = self.tokenizer(text2, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        if ids1.size(1) == 0 or ids2.size(1) == 0:
            return 0.0

        with torch.no_grad(), torch.amp.autocast('cuda'):
            embs1 = self.model.wte(ids1).squeeze(0)
            embs2 = self.model.wte(ids2).squeeze(0)
            embs1_norm = F.normalize(embs1, p=2, dim=1)
            embs2_norm = F.normalize(embs2, p=2, dim=1)
            sim_matrix = torch.matmul(embs1_norm, embs2_norm.T)
            precision  = sim_matrix.max(dim=1)[0].mean().item()
            recall     = sim_matrix.max(dim=0)[0].mean().item()
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

    def beam_search(self, prompt, beam_width=3, max_len=15, guided=False):
        self.model.eval()
        initial_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)[0]
        beams = [(initial_ids, 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    _, logits         = self.model(seq.unsqueeze(0))
                    next_token_logits = logits[0, -1, :]
                    log_probs         = F.log_softmax(next_token_logits, dim=-1)

                top_probs, top_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    new_seq    = torch.cat([seq, top_ids[i].unsqueeze(0)])
                    step_score = top_probs[i].item()

                    if guided:
                        gen_text = self.tokenizer.decode(new_seq[len(initial_ids):], skip_special_tokens=True).strip()
                        bert_score      = self.compute_bertscore(prompt, gen_text) if gen_text else 0.0
                        composite_score = step_score + 0.5 * bert_score
                        new_beams.append((new_seq, score + composite_score))
                    else:
                        new_beams.append((new_seq, score + step_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if beams[0][0][-1].item() == self.tokenizer.eos_token_id:
                break

        best_seq  = beams[0][0][len(initial_ids):]
        decoded   = self.tokenizer.decode(best_seq, skip_special_tokens=True).strip()
        return decoded if decoded else "unknown"


def run_eval(pipeline, dataset, title_string):
    task2_correct = task3_correct = task4_correct = 0
    total = len(dataset)

    print(f"\nEvaluating: {title_string}")
    for i in tqdm(range(total), desc="Inference"):
        item       = dataset[i]
        fact, stem = item["raw_fact"], item["raw_stem"]
        choices    = item["raw_choices"]
        true_label = item["label_idx"]

        prompt = f"{fact} {stem}"

        probs = [pipeline.sequence_probability(prompt, c) for c in choices]
        pred2 = probs.index(max(probs))
        if pred2 == true_label:
            task2_correct += 1

        vanilla_gen = pipeline.beam_search(prompt, guided=False)
        guided_gen  = pipeline.beam_search(prompt, guided=True)

        vanilla_scores = [pipeline.compute_bertscore(vanilla_gen, c) for c in choices]
        guided_scores  = [pipeline.compute_bertscore(guided_gen,  c) for c in choices]

        pred3 = vanilla_scores.index(max(vanilla_scores))
        pred4 = guided_scores.index(max(guided_scores))

        if pred3 == true_label: task3_correct += 1
        if pred4 == true_label: task4_correct += 1

    print(f"\nResults for {title_string}:")
    print(f"Task 2 Accuracy: {task2_correct / total * 100:.2f}%")
    print(f"Task 3 Accuracy: {task3_correct / total * 100:.2f}%")
    print(f"Task 4 Accuracy: {task4_correct / total * 100:.2f}%")


def main():
    pipeline = AsterPipeline()

    print("Loading data splits...")
    # We only need the validation and test sets for the required report metrics
    valid_dataset = AsterOBQADataset('obqa/obqa.valid.txt', pipeline.tokenizer)
    test_dataset  = AsterOBQADataset('obqa/obqa.test.txt',  pipeline.tokenizer)

    # --- PART 1: ZERO SHOT ---
    print("\n" + "="*40)
    print("=== STARTING ZERO-SHOT EVALUATIONS ===")
    print("="*40)
    
    # ONLY evaluating the 500-item sets (Takes ~15 mins each)
    run_eval(pipeline, valid_dataset, "Validation Set (Zero-Shot)")
    run_eval(pipeline, test_dataset, "Test Set (Zero-Shot)")

    # --- PART 2: FINE TUNED ---
    print("\n" + "="*40)
    print("=== STARTING FINE-TUNED EVALUATIONS ===")
    print("="*40)
    
    checkpoint_path = "aster_finetuned.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        return

    print(f"Loading weights from {checkpoint_path}...")
    pipeline.model.load_state_dict(torch.load(checkpoint_path, map_location=pipeline.device))

    # ONLY evaluating the 500-item sets
    run_eval(pipeline, valid_dataset, "Validation Set (Fine-Tuned)")
    run_eval(pipeline, test_dataset, "Test Set (Fine-Tuned)")


if __name__ == "__main__":
    main()