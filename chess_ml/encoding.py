import torch
import torch.nn.functional as F

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
    return torch.tensor(encoding, dtype=torch.int8)

def fen_to_target(fen): # encodes each square with numbers 0-12 representing empty, PRNBQK, prnbqk
    encoding = []
    pieces = 'PRNBQKprnbqk'
    mapping = {pieces[i] : i+1 for i in range(12)}
    for char in fen:
        if char in pieces:
            encoding.append(mapping[char])
        elif char in '12345678':
            encoding.extend([0 for i in range(int(char))])
    return torch.tensor(encoding, dtype=torch.int8)

def output_to_fen(output): # 64*13-entry tensor to fen
    predictions = output.argmax(dim = 1)
    pieces = 'PRNBQKprnbqk'
    mapping = {i+1 : pieces[i] for i in range(12)}
    fen = ''
    run_length = 0
    for i in range(64):
        if i % 8 == 0 and i != 0:
            if run_length > 0:
                fen += str(run_length)
                run_length = 0
            fen += '/'
        if predictions[i] == 0:
            run_length += 1
        else:
            if run_length > 0:
                fen += str(run_length)
                run_length = 0
            fen += mapping[int(predictions[i])]
    if run_length > 0:
        fen += str(run_length)
    return fen

def output_to_probabilities(output): # softmax
    return F.softmax(output, dim = 1)