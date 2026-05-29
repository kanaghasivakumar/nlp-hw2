import re

def extract_examples(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Could not find {filename}. Make sure you are in the right directory.")
        return
    
    # Split the document into individual evaluation blocks
    blocks = content.split("Fact: ")[1:]
    
    v_correct, v_incorrect = [], []
    g_correct, g_incorrect = [], []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 6: 
            continue
            
        fact = lines[0].strip()
        stem = lines[1].replace("Stem:", "").strip()
        choices = lines[2].replace("Choices:", "").strip()
        
        # Safely extract the Gold label and Predictions
        gold_match = re.search(r"Gold:\s*(\d+)", lines[3])
        if not gold_match: continue
        gold = int(gold_match.group(1))
        
        vanilla_line = lines[4]
        guided_line = lines[5]
        
        v_pred_match = re.search(r"Pred:\s*(\d+)", vanilla_line)
        g_pred_match = re.search(r"Pred:\s*(\d+)", guided_line)
        
        if not v_pred_match or not g_pred_match: continue
        v_pred = int(v_pred_match.group(1))
        g_pred = int(g_pred_match.group(1))
        
        # Format the output block
        out_str = f"Fact: {fact}\nStem: {stem}\nChoices: {choices}\nGold Answer: {gold}\n"
        
        # Sort Vanilla Examples
        if len(v_correct) < 5 and v_pred == gold:
            v_correct.append(out_str + f"Predicted: {v_pred}\n{vanilla_line}\n")
        elif len(v_incorrect) < 5 and v_pred != gold:
            v_incorrect.append(out_str + f"Predicted: {v_pred}\n{vanilla_line}\n")
            
        # Sort Guided Examples
        if len(g_correct) < 5 and g_pred == gold:
            g_correct.append(out_str + f"Predicted: {g_pred}\n{guided_line}\n")
        elif len(g_incorrect) < 5 and g_pred != gold:
            g_incorrect.append(out_str + f"Predicted: {g_pred}\n{guided_line}\n")
            
        # Break early if we have all 20 required examples
        if len(v_correct) == 5 and len(v_incorrect) == 5 and len(g_correct) == 5 and len(g_incorrect) == 5:
            break

    print("="*50 + "\nTASK 3: VANILLA BEAM SEARCH (5 CORRECT)\n" + "="*50)
    print("\n---\n".join(v_correct))
    
    print("\n" + "="*50 + "\nTASK 3: VANILLA BEAM SEARCH (5 INCORRECT)\n" + "="*50)
    print("\n---\n".join(v_incorrect))
    
    print("\n" + "="*50 + "\nTASK 4: GUIDED BEAM SEARCH (5 CORRECT)\n" + "="*50)
    print("\n---\n".join(g_correct))
    
    print("\n" + "="*50 + "\nTASK 4: GUIDED BEAM SEARCH (5 INCORRECT)\n" + "="*50)
    print("\n---\n".join(g_incorrect))

if __name__ == "__main__":
    # Point this to whichever log file you want to pull from (validation is standard for the report)
    extract_examples("finetuned_valid_logs.txt")