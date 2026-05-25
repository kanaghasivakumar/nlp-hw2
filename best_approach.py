import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class OBQADataset(Dataset):
    """
    Parses the local text files and formats the inputs for BERT.
    """
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = []
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        
        with open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split('|')
                fact = tokens[0]
                stem = tokens[1]
                choices = [tokens[2], tokens[3], tokens[4], tokens[5]]
                answer = tokens[6]
                
                choice_input_ids = []
                choice_attention_masks = []
                
                for choice in choices:
                    text = f"[CLS] {fact} {stem} {choice} [SEP]"
                    encoded = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        add_special_tokens=False
                    )
                    choice_input_ids.append(encoded['input_ids'])
                    choice_attention_masks.append(encoded['attention_mask'])
                
                self.data.append({
                    "input_ids": torch.tensor(choice_input_ids),
                    "attention_mask": torch.tensor(choice_attention_masks),
                    "label": torch.tensor(self.label_map[answer])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BertQAClassifier(nn.Module):
    """
    Wraps the standard BERT base model with a custom classification head.
    """
    def __init__(self, checkpoint="google-bert/bert-base-uncased"):
        super(BertQAClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        batch_size, num_choices, seq_length = input_ids.shape
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output 
        logits = self.classifier(cls_output)
        reshaped_logits = logits.view(-1, num_choices)
        return reshaped_logits

def plot_losses(train_losses, val_losses):
    """
    Generates and saves a plot of the training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('bert_loss_curve.png')
    plt.close()

def train_and_evaluate():
    """
    Executes the training loop with early stopping and saves the best model weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    train_dataset = OBQADataset('obqa/obqa.train.txt', tokenizer)
    valid_dataset = OBQADataset('obqa/obqa.valid.txt', tokenizer)
    test_dataset = OBQADataset('obqa/obqa.test.txt', tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BertQAClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    zero_shot_loss, zero_shot_acc = evaluate(model, valid_loader)
    print(f"Zero-shot Validation Accuracy: {zero_shot_acc * 100:.2f}%")

    epochs = 10
    patience = 2
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, valid_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "bert_best_model.pt")
            print("Saved new best model weights.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    plot_losses(train_losses, val_losses)
    
    model.load_state_dict(torch.load("bert_best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy (Fine-tuned): {test_acc * 100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()