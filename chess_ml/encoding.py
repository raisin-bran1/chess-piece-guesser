# fen to 64*3-entry tensor for model input
# fen to 64-entry tensor for target output
# model output (64*13-entry tensor) to fen 

import torch

def fen_to_input(fen): # encodes each square as 1,0,0 / 0,1,0 / 0,0,1 for empty / white / black
    encoding = []
    for char in fen:
        if char in 'PRNBQK':
            encoding.extend([0, 1, 0])
        elif char in 'prnbqk':
            encoding.extend([0, 0, 1])
        elif char in '12345678':
            for i in range(int(char)):
                encoding.extend([1, 0, 0])
    print(encoding)
    return torch.tensor(encoding)

def fen_to_target(fen): # encodes each square with numbers 0-12 representing empty, PRNBQK, prnbqk
    encoding = []
    pieces = 'PRNBQKprnbqk'
    mapping = {pieces[i] : i+1 for i in range(12)}
    for char in fen:
        if char in pieces:
            encoding.append(mapping[char])
        elif char in '12345678':
            encoding.extend([0 for i in range(int(char))])
    print(encoding)
    return torch.tensor(encoding)

def output_to_fen(output):
    pass # TO DO