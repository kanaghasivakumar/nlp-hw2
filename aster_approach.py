import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
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

                prompt_text = f"{fact} {stem} The answer is"
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
                    "input_ids":      torch.tensor(all_input_ids),       # [4, max_length]
                    "attention_mask": torch.tensor(all_attention_mask),  # [4, max_length]
                    "labels":         torch.tensor(all_labels),          # [4, max_length]
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

    def fine_tune(self, train_loader, valid_loader, epochs=10, patience=4, save_name="aster_finetuned.pt"):

        for name, param in self.model.named_parameters():
            param.requires_grad = True
        for name, param in self.model.named_parameters():
            if "wpe" in name:
                param.requires_grad = False

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable params: {len(trainable)}")

        optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss    = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0 

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1})"):
                B         = batch["input_ids"].size(0)
                label_idx = batch["label_idx"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels    = batch["labels"].to(self.device)

                input_ids_flat = input_ids.view(B * 4, -1)
                attn_flat      = attn_mask.view(B * 4, -1)
                labels_flat    = labels.view(B * 4, -1)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    _, logits    = self.model(input_ids_flat, attention_mask=attn_flat)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_flat[..., 1:].contiguous()

                    losses_flat = torch.zeros(B * 4, device=self.device)
                    for k in range(B * 4):
                        n_toks = (shift_labels[k] != -100).sum().clamp(min=1)
                        losses_flat[k] = ce(
                            shift_logits[k].view(-1, shift_logits.size(-1)),
                            shift_labels[k].view(-1)
                        ) / n_toks

                    losses = losses_flat.view(B, 4)
                    loss = losses[torch.arange(B), label_idx].mean()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    B         = batch["input_ids"].size(0)
                    label_idx = batch["label_idx"].to(self.device)
                    input_ids = batch["input_ids"].to(self.device)
                    attn_mask = batch["attention_mask"].to(self.device)
                    labels    = batch["labels"].to(self.device)

                    input_ids_flat = input_ids.view(B * 4, -1)
                    attn_flat      = attn_mask.view(B * 4, -1)
                    labels_flat    = labels.view(B * 4, -1)

                    with torch.amp.autocast('cuda'):
                        _, logits    = self.model(input_ids_flat, attention_mask=attn_flat)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels_flat[..., 1:].contiguous()

                        losses_flat = torch.zeros(B * 4, device=self.device)
                        for k in range(B * 4):
                            n_toks = (shift_labels[k] != -100).sum().clamp(min=1)
                            losses_flat[k] = ce(
                                shift_logits[k].view(-1, shift_logits.size(-1)),
                                shift_labels[k].view(-1)
                            ) / n_toks

                        losses = losses_flat.view(B, 4)
                        loss = losses[torch.arange(B), label_idx].mean()

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(valid_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            scheduler.step()

            print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_name)
                print(f"  --> Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        self._plot_losses(train_losses, val_losses, save_name.replace('.pt', '_loss.png'))

    def _plot_losses(self, train_losses, val_losses, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses,   label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def sequence_probability(self, prompt, choice):
        """Score only choice tokens conditioned on prompt. Length-penalized."""
        self.model.eval()
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)
        choice_ids = self.tokenizer(
            " " + choice, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)
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
        initial_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)[0]
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
                        gen_text = self.tokenizer.decode(
                            new_seq[len(initial_ids):], skip_special_tokens=True
                        ).strip()
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


def execute_evaluation(pipeline, dataset, output_log="generation_logs.txt",
                       split_name="", skip_beam=False):
    task2_correct = task3_correct = task4_correct = 0
    total = len(dataset)
    logs  = []

    for i in tqdm(range(total), desc=f"Evaluating {split_name}"):
        item       = dataset[i]
        fact, stem = item["raw_fact"], item["raw_stem"]
        choices    = item["raw_choices"]
        true_label = item["label_idx"]

        prompt = f"{fact} {stem} The answer is"

        probs = [pipeline.sequence_probability(prompt, c) for c in choices]
        pred2 = probs.index(max(probs))
        if pred2 == true_label:
            task2_correct += 1

        if not skip_beam:
            vanilla_gen = pipeline.beam_search(prompt, guided=False)
            guided_gen  = pipeline.beam_search(prompt, guided=True)

            vanilla_scores = [pipeline.compute_bertscore(vanilla_gen, c) for c in choices]
            guided_scores  = [pipeline.compute_bertscore(guided_gen,  c) for c in choices]

            pred3 = vanilla_scores.index(max(vanilla_scores))
            pred4 = guided_scores.index(max(guided_scores))

            if pred3 == true_label: task3_correct += 1
            if pred4 == true_label: task4_correct += 1

            logs.append(f"Fact: {fact}\nStem: {stem}\nChoices: {choices}\nGold: {true_label}")
            logs.append(f"Vanilla Gen: {vanilla_gen} | Pred: {pred3} | Scores: {vanilla_scores}")
            logs.append(f"Guided Gen:  {guided_gen}  | Pred: {pred4} | Scores: {guided_scores}\n")

    t2 = task2_correct / total * 100
    t3 = task3_correct / total * 100 if not skip_beam else None
    t4 = task4_correct / total * 100 if not skip_beam else None

    print(f"\n{'='*55}")
    print(f"  {split_name}  (n={total})")
    print(f"{'='*55}")
    print(f"  Task 2 (Seq Prob):        {t2:.2f}%  [Prof: 27.6% / 57.6%]")
    if not skip_beam:
        print(f"  Task 3 (Vanilla Beam):    {t3:.2f}%  [Prof: 26.0% / 43.8%]")
        print(f"  Task 4 (Guided Beam):     {t4:.2f}%  [Prof: ~26%  / 47.6%]")
    print(f"{'='*55}\n")

    with open(output_log, 'w') as f:
        f.write("\n".join(logs))

    return t2, t3, t4

def main():
    pipeline = AsterPipeline()

    valid_dataset = AsterOBQADataset('obqa/obqa.valid.txt', pipeline.tokenizer)
    test_dataset  = AsterOBQADataset('obqa/obqa.test.txt',  pipeline.tokenizer)

    print("\n>>> ZERO-SHOT EVALUATION")
    execute_evaluation(pipeline, valid_dataset, "zero_shot_valid_logs.txt",
                       split_name="Validation (zero-shot)", skip_beam=True)
    execute_evaluation(pipeline, test_dataset,  "zero_shot_test_logs.txt",
                       split_name="Test (zero-shot)", skip_beam=True)

    train_dataset = AsterOBQADataset('obqa/obqa.train.txt', pipeline.tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True,
                               num_workers=os.cpu_count(), pin_memory=True, prefetch_factor=2)
    valid_loader  = DataLoader(valid_dataset, batch_size=16,
                               num_workers=os.cpu_count(), pin_memory=True)

    print("\n>>> FINE-TUNING")
    pipeline.fine_tune(train_loader, valid_loader, save_name="aster_finetuned.pt")

    print("\n>>> Loading best fine-tuned checkpoint...")
    pipeline.model.load_state_dict(
        torch.load("aster_finetuned.pt", map_location=pipeline.device)
    )

    print("\n>>> FINE-TUNED EVALUATION")
    execute_evaluation(pipeline, valid_dataset, "finetuned_valid_logs.txt",
                       split_name="Validation (fine-tuned)", skip_beam=False)
    execute_evaluation(pipeline, test_dataset,  "finetuned_test_logs.txt",
                       split_name="Test (fine-tuned)", skip_beam=False)


if __name__ == "__main__":
    main()