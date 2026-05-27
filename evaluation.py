import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

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

                self.data.append({
                    "raw_fact":    fact,
                    "raw_stem":    stem,
                    "raw_choices": choices,
                    "label_idx":   self.label_map[answer_key]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_pipeline(checkpoint_path="aster_finetuned.pt",
                  tokenizer_dir="saved/tokenizer"):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    state_dict = load_model_best_state_dict(checkpoint_path)
    vocab_size  = int(state_dict["wte.weight"].shape[0])

    model = TransformerGPT(vocab_size=vocab_size, d_model=1024, n_layers=16,
                           heads=16, seqlen=1024, d_ff=4096)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def sequence_probability(model, tokenizer, device, prompt, choice):
    """Professor-style manual softmax scoring, choice tokens only."""
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    choice_ids = tokenizer(
        " " + choice, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, choice_ids], dim=1)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        _, preds = model(input_ids)

        prompt_len    = prompt_ids.size(1)
        n_choice_toks = choice_ids.size(1)
        seq_log_prob  = 0.0

        for i in range(prompt_len - 1, prompt_len + n_choice_toks - 1):
            target_token = input_ids[0, i + 1].item()
            logit_vec    = preds[0, i, :]
            logit_vec    = logit_vec - logit_vec.max()   # numerical stability
            exp_logits   = torch.exp(logit_vec)
            prob         = exp_logits[target_token] / exp_logits.sum()
            seq_log_prob += torch.log(prob + 1e-12).item()

    return seq_log_prob / (n_choice_toks ** 0.7)


def evaluate(model, tokenizer, device, dataset, split_name=""):
    correct = 0
    total   = len(dataset)

    for i in tqdm(range(total), desc=f"Evaluating {split_name}"):
        item       = dataset[i]
        fact, stem = item["raw_fact"], item["raw_stem"]
        choices    = item["raw_choices"]
        true_label = item["label_idx"]

        prompt = f"{fact} {stem}"
        probs  = [sequence_probability(model, tokenizer, device, prompt, c) for c in choices]
        pred   = probs.index(max(probs))
        if pred == true_label:
            correct += 1

    acc = correct / total * 100
    print(f"\n{'='*55}")
    print(f"  {split_name}  (n={total})")
    print(f"{'='*55}")
    print(f"  Task 2 (Seq Prob): {acc:.2f}%  [Prof: 27.6% / 57.6%]")
    print(f"{'='*55}\n")
    return acc


def main():
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "aster_finetuned.pt"
    print(f"Loading checkpoint: {checkpoint}")

    model, tokenizer, device = load_pipeline(checkpoint)

    valid_dataset = AsterOBQADataset('obqa/obqa.valid.txt', tokenizer)
    test_dataset  = AsterOBQADataset('obqa/obqa.test.txt',  tokenizer)

    evaluate(model, tokenizer, device, valid_dataset, "Validation (fine-tuned)")
    evaluate(model, tokenizer, device, test_dataset,  "Test (fine-tuned)")


if __name__ == "__main__":
    main()