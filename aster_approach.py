import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from Aster import TransformerGPT, load_model_best_state_dict


class AsterOBQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = []
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        with open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                tokens = line.split('|')
                fact = tokens[0]
                stem = tokens[1]
                choices = {"A": tokens[2], "B": tokens[3], "C": tokens[4], "D": tokens[5]}
                answer_key = tokens[6]

                correct_choice = choices[answer_key]
                prompt_text = f"{fact}\n{stem}\n"
                choice_text = correct_choice + (tokenizer.eos_token if tokenizer.eos_token else "")

                prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
                choice_encoded = tokenizer(choice_text, add_special_tokens=False)['input_ids']

                input_ids = prompt_encoded + choice_encoded
                labels = input_ids.copy()
                attention_mask = [1] * len(input_ids)

                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([tokenizer.pad_token_id] * pad_len)
                    labels.extend([-100] * pad_len)
                    attention_mask.extend([0] * pad_len)
                else:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                    attention_mask = attention_mask[:max_length]

                for i in range(len(prompt_encoded)):
                    labels[i] = -100

                self.data.append({
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels),
                    "raw_fact": fact,
                    "raw_stem": stem,
                    "raw_choices": list(choices.values()),
                    "label_idx": self.label_map[answer_key]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AsterPipeline:
    def __init__(self, tokenizer_dir="saved/tokenizer", model_path="saved/model_best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"

        self.model_path = model_path 
        state_dict = load_model_best_state_dict(model_path)
        vocab_size = int(state_dict["wte.weight"].shape[0])

        self.model = TransformerGPT(vocab_size=vocab_size, d_model=1024, n_layers=16,
                                    heads=16, seqlen=1024, d_ff=4096)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def fine_tune(self, train_loader, valid_loader, epochs=5, patience=2, save_name="aster_finetuned.pt"):
        for name, param in self.model.named_parameters():
            param.requires_grad = True

        for name, param in self.model.named_parameters():
            if "wpe" in name: 
                param.requires_grad = False
            for i in range(6): 
                if f".{i}." in name:
                    param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Trainable param groups: {len(trainable_names)} — first few: {trainable_names[:4]}")

        optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    _, logits = self.model(input_ids, attention_mask=attention_mask)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  # gradient clipping
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        _, logits = self.model(input_ids, attention_mask=attention_mask)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(valid_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def sequence_probability(self, prompt, choice):
        self.model.eval()
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        choice_ids = self.tokenizer(choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        input_ids = torch.cat([prompt_ids, choice_ids], dim=1)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            _, logits = self.model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)

            seq_log_prob = 0.0
            prompt_len = prompt_ids.size(1)

            for i in range(prompt_len - 1, input_ids.size(1) - 1):
                target_token = input_ids[0, i + 1]
                seq_log_prob += log_probs[0, i, target_token].item()

        n_choice_tokens = choice_ids.size(1)
        return seq_log_prob / (n_choice_tokens ** 0.7)

    def compute_bertscore(self, text1, text2):
        ids1 = self.tokenizer(text1, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        ids2 = self.tokenizer(text2, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            embs1 = self.model.wte(ids1).squeeze(0)
            embs2 = self.model.wte(ids2).squeeze(0)

            embs1_norm = F.normalize(embs1, p=2, dim=1)
            embs2_norm = F.normalize(embs2, p=2, dim=1)

            sim_matrix = torch.matmul(embs1_norm, embs2_norm.T)

            recall = sim_matrix.max(dim=0)[0].mean().item()
            precision = sim_matrix.max(dim=1)[0].mean().item()

            if precision + recall == 0:
                return 0.0
            f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def beam_search(self, prompt, beam_width=3, max_len=15, guided=False):
        self.model.eval()
        initial_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)[0]
        beams = [(initial_ids, 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    _, logits = self.model(seq.unsqueeze(0))
                    next_token_logits = logits[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)

                top_probs, top_ids = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat([seq, top_ids[i].unsqueeze(0)])
                    step_score = top_probs[i].item()

                    if guided:
                        gen_text = self.tokenizer.decode(new_seq[len(initial_ids):], skip_special_tokens=True)
                        bert_score = self.compute_bertscore(prompt, gen_text)
                        composite_score = step_score + (0.5 * bert_score)
                        new_beams.append((new_seq, score + composite_score))
                    else:
                        new_beams.append((new_seq, score + step_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            if beams[0][0][-1].item() == self.tokenizer.eos_token_id:
                break

        best_seq = beams[0][0][len(initial_ids):]
        return self.tokenizer.decode(best_seq, skip_special_tokens=True)


def execute_evaluation(pipeline, dataset, output_log="generation_logs.txt", split_name="", skip_beam=False):
    task2_correct = 0
    task3_correct = 0
    task4_correct = 0
    total = len(dataset)
    logs = []

    for i in tqdm(range(total), desc=f"Evaluating {split_name}"):
        item = dataset[i]
        fact, stem = item["raw_fact"], item["raw_stem"]
        choices = item["raw_choices"]
        true_label = item["label_idx"]
        prompt = f"{fact}\n{stem}\n"

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
            logs.append(f"Guided Gen: {guided_gen} | Pred: {pred4} | Scores: {guided_scores}\n")

    t2 = task2_correct / total * 100
    t3 = task3_correct / total * 100 if not skip_beam else None
    t4 = task4_correct / total * 100 if not skip_beam else None

    print(f"\n{'='*50}")
    print(f"Results on {split_name} set (n={total})")
    print(f"{'='*50}")
    print(f"  Task 2 (Sequence Prob):       {t2:.2f}%  [Prof zero-shot: 27.6%]")
    if not skip_beam:
        print(f"  Task 3 (Vanilla Beam Search): {t3:.2f}%  [Prof zero-shot: 26.0%]")
        print(f"  Task 4 (Guided Beam Search):  {t4:.2f}%  [Prof zero-shot: ~26%]")
    print(f"{'='*50}\n")

    with open(output_log, 'w') as f:
        f.write("\n".join(logs))

    return t2, t3, t4


def main():
    pipeline = AsterPipeline()

    valid_dataset = AsterOBQADataset('obqa/obqa.valid.txt', pipeline.tokenizer)
    test_dataset  = AsterOBQADataset('obqa/obqa.test.txt',  pipeline.tokenizer)

    print("\n>>> ZERO-SHOT EVALUATION")
    execute_evaluation(pipeline, valid_dataset, "zero_shot_valid_logs.txt", split_name="Validation (zero-shot)", skip_beam=True)
    execute_evaluation(pipeline, test_dataset,  "zero_shot_test_logs.txt",  split_name="Test (zero-shot)",       skip_beam=True)

    train_dataset = AsterOBQADataset('obqa/obqa.train.txt', pipeline.tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2
    )
    valid_loader = DataLoader(valid_dataset, batch_size=8, num_workers=os.cpu_count(), pin_memory=True)

    print("\n>>> FINE-TUNING")
    pipeline.fine_tune(train_loader, valid_loader, save_name="aster_finetuned.pt")

    print("\n>>> Loading best fine-tuned checkpoint.")
    pipeline.model.load_state_dict(torch.load("aster_finetuned.pt", map_location=pipeline.device))

    print("\n>>> FINE-TUNED EVALUATION")
    execute_evaluation(pipeline, valid_dataset, "finetuned_valid_logs.txt", split_name="Validation (fine-tuned)")
    execute_evaluation(pipeline, test_dataset,  "finetuned_test_logs.txt",  split_name="Test (fine-tuned)")


if __name__ == "__main__":
    main()