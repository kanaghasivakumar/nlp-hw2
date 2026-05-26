import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

from Aster import TransformerGPT, load_model_best_state_dict

class AsterOBQADataset(Dataset):
    """
    Parses local text files and formats inputs for Aster's autoregressive training.
    """
    def __init__(self, file_path, tokenizer, max_length=256, task="classification"):
        self.data = []
        with open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split('|')
                fact = tokens[0]
                stem = tokens[1]
                choices = {"A": tokens[2], "B": tokens[3], "C": tokens[4], "D": tokens[5]}
                answer_key = tokens[6]
               
                if task == "classification":
                    correct_choice = choices[answer_key]
                    text = f"{fact} {stem} {correct_choice}"
                else:
                    correct_choice = choices[answer_key]
                    text = f"{fact} {stem} [ANSWER] {correct_choice}"

                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
               
                self.data.append({
                    "input_ids": encoded['input_ids'].squeeze(0),
                    "attention_mask": encoded['attention_mask'].squeeze(0),
                    "raw_fact": fact,
                    "raw_stem": stem,
                    "raw_choices": list(choices.values()),
                    "label_idx": ord(answer_key) - ord('A')
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AsterPipeline:
    """
    Manages fine-tuning and inference for Tasks 2, 3, and 4.
    """
    def __init__(self, tokenizer_dir="saved/tokenizer", model_path="saved/model_best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"
       
        state_dict = load_model_best_state_dict(model_path)
        vocab_size = int(state_dict["wte.weight"].shape[0])
       
        self.model = TransformerGPT(vocab_size=vocab_size, d_model=1024, n_layers=16,
                                    heads=16, seqlen=1024, d_ff=4096)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def fine_tune(self, train_loader, valid_loader, epochs=5, patience=2, save_name="aster_finetuned.pt"):
        """
        Executes autoregressive fine-tuning with early stopping, mixed precision, and loss plotting.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
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

                optimizer.zero_grad()
               
                with torch.amp.autocast('cuda'):
                    _, logits = self.model(input_ids, attention_mask=attention_mask)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
               
                scaler.scale(loss).backward()
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
                   
                    with torch.amp.autocast('cuda'):
                        _, logits = self.model(input_ids, attention_mask=attention_mask)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                       
                    total_val_loss += loss.item()
           
            avg_val_loss = total_val_loss / len(valid_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
           
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_name)
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

    def sequence_probability(self, text):
        """
        Calculates the log probability of a specific sequence for Task 2 using mixed precision.
        """
        self.model.eval()
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
       
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _, logits = self.model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
           
            seq_log_prob = 0.0
            for i in range(input_ids.size(1) - 1):
                target_token = input_ids[0, i+1]
                seq_log_prob += log_probs[0, i, target_token].item()
               
        return seq_log_prob

    def compute_bertscore(self, text1, text2):
        """
        Calculates similarity using Aster's word embeddings.
        """
        ids1 = self.tokenizer(text1, return_tensors="pt")["input_ids"].to(self.device)
        ids2 = self.tokenizer(text2, return_tensors="pt")["input_ids"].to(self.device)
       
        with torch.amp.autocast('cuda'):
            embs1 = self.model.wte(ids1).squeeze(0)
            embs2 = self.model.wte(ids2).squeeze(0)
           
            vec1 = embs1.mean(dim=0)
            vec2 = embs2.mean(dim=0)
           
            similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
           
        return similarity.item()

    def beam_search(self, prompt, beam_width=3, max_len=15, guided=False):
        """
        Generates text using vanilla or guided beam search.
        """
        self.model.eval()
        initial_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)[0]
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


def execute_evaluation(pipeline, dataset, output_log="generation_logs.txt"):
    """
    Runs inference for Tasks 2, 3, and 4 on the provided dataset and logs generations.
    """
    task2_correct = 0
    task3_correct = 0
    task4_correct = 0
    total = len(dataset)
    logs = []

    for i in tqdm(range(total), desc="Evaluating"):
        item = dataset[i]
        fact, stem = item["raw_fact"], item["raw_stem"]
        choices = item["raw_choices"]
        true_label = item["label_idx"]
       
        probs = [pipeline.sequence_probability(f"{fact} {stem} {c}") for c in choices]
        pred2 = probs.index(max(probs))
        if pred2 == true_label:
            task2_correct += 1
           
        prompt = f"{fact} {stem} [ANSWER]"
        vanilla_gen = pipeline.beam_search(prompt, guided=False)
        guided_gen = pipeline.beam_search(prompt, guided=True)
       
        vanilla_scores = [pipeline.compute_bertscore(vanilla_gen, c) for c in choices]
        guided_scores = [pipeline.compute_bertscore(guided_gen, c) for c in choices]
       
        pred3 = vanilla_scores.index(max(vanilla_scores))
        pred4 = guided_scores.index(max(guided_scores))
       
        if pred3 == true_label: task3_correct += 1
        if pred4 == true_label: task4_correct += 1
       
        logs.append(f"Fact: {fact}\nStem: {stem}\nChoices: {choices}\nGold: {true_label}")
        logs.append(f"Vanilla Gen: {vanilla_gen} | Pred: {pred3} | Scores: {vanilla_scores}")
        logs.append(f"Guided Gen: {guided_gen} | Pred: {pred4} | Scores: {guided_scores}\n")
       
    print(f"Task 2 Accuracy: {task2_correct / total * 100:.2f}%")
    print(f"Task 3 Accuracy: {task3_correct / total * 100:.2f}%")
    print(f"Task 4 Accuracy: {task4_correct / total * 100:.2f}%")
   
    with open(output_log, 'w') as f:
        f.write("\n".join(logs))


def main():
    pipeline = AsterPipeline()
   
    valid_dataset = AsterOBQADataset('obqa/obqa.valid.txt', pipeline.tokenizer)
    print("Running Zero-Shot Evaluation.")
    execute_evaluation(pipeline, valid_dataset, "zero_shot_logs.txt")
   
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
   
    print("Fine-tuning Aster.")
    pipeline.fine_tune(train_loader, valid_loader)
   
    pipeline.model.load_state_dict(torch.load("aster_finetuned.pt"))
    test_dataset = AsterOBQADataset('obqa/obqa.test.txt', pipeline.tokenizer)
   
    print("Running Fine-Tuned Evaluation on Test Set.")
    execute_evaluation(pipeline, test_dataset, "finetuned_logs.txt")

if __name__ == "__main__":
    main()