import os
import sys
import yaml
import torch
import argparse
import numpy as np
from vocabulary import TOKENS
from torch.optim import Adam
from transformer import Transformer
from lightning.fabric import Fabric
from torch import set_float32_matmul_precision


def beam_search(model, input_sequence, vocab, beam_width, device='cuda'):
    candidates = [[input_sequence, 0.0]]
    output_text = []
    input_sequence = torch.LongTensor([input_sequence]).to(device)
    encoded_x = model.encode(input_sequence) 
    
    for i in range(512):
        new_candidates = []
        for candidate, score in candidates:
            input_sequence = torch.LongTensor([candidate]).to(device)
            output = model.decode(input_sequence, encoded_x)
            logits = torch.nn.functional.softmax(output, dim=-1).detach().cpu().numpy()[0, -1, :]
            log_probs = np.log(np.exp(logits) / np.sum(np.exp(logits)))
            top_k_indices = np.argsort(log_probs)[-beam_width:]
            for index in top_k_indices:
                new_candidate = candidate + [index]
                new_score = score + log_probs[index]
                new_candidates.append([new_candidate, new_score])
                
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_width]
        if not candidates:
            break

    for candidate, _ in candidates:
        sequence = ""
        for index in candidate:
            if index == TOKENS.index(b'PAD'):
                sequence += "p"
            elif index == TOKENS.index(b'MASK'):
                sequence += "m"
            elif index != TOKENS.index(b'START') and index != TOKENS.index(b'END'):
                sequence += chr(vocab[index])
            elif index == TOKENS.index(b'END'):
                break 
        output_text.append(sequence)
    return output_text


def main(args, config):
    set_float32_matmul_precision('medium')

    # Init Fabric
    fabric = Fabric(accelerator="gpu", 
                    devices=config["devices"],
                    precision="16-mixed",
                    strategy="deepspeed")
    fabric.launch()

    # Model
    model = Transformer(args.num_tokens)
    optimizer = Adam(model.parameters(), config["lr"])
    model, optimizer = fabric.setup(model, optimizer)

    # Load pretrained model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = {"model": model, "optimizer": optimizer}
    fabric.load(args.checkpoint, state)

    # Generate sequences
    input_sequence = [0] 
    output_sequences = []
    fabric.print("Generating from", args.checkpoint, "...\n")
    for _ in range(args.n):
        sequences = beam_search(model, input_sequence, TOKENS, args.k)
        output_sequences.extend(sequences)
    
    # Writing sequences to FASTA file
    output_file = os.path.join(config["out_dir"], "generated.fasta")
    with open(output_file, "wt") as f:
        for i, sequence in enumerate(output_sequences, start=1):
            identifier = f"Sequence_{i}"
            f.write(f">{identifier}\n")
            f.write(f"{sequence}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True, help="number of sequences to generate")
    parser.add_argument("-checkpoint", type=str, required=True, help="model checkpoint path")
    parser.add_argument("-k", type=int, default=3, help="beam width")
    parser.add_argument("-num_tokens", default=29, type=int, help="number of tokens of the model")
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    with open(sys.path[0] + '/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(args, config)
