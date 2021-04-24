import torch
import torch.nn.functional as F
import random

ops = {
    'add':  lambda x,y: x+y, 
    'xor':  lambda x,y: x^y, 
    'and':  lambda x,y: x&y, 
    'or':   lambda x,y: x|y
}
# [0, 1, ... , e, f]
hex_vocab = list(map(str, range(10))) + [chr(i) for i in range(ord('a'), ord('f')+1)]

def format_targ(targ):
    """ Truncate or padd with 0s """
    targ = targ[2:]
    while len(targ) > 16:
        targ = targ[1:]
    while len(targ) < 16:
        targ = '0' + targ
    return targ

def byte_tokenize(word):
    assert len(word) == 16
    ##print(word)
    #hi
    return torch.tensor([int(word[i:i+2], 16) for i in range(0, len(word), 2)])

def op_tokenize(op_name):
    assert op_name in ops.keys()
    return torch.tensor(list(ops.keys()).index(op_name))

def one_hot(targ):
    return F.one_hot(targ, num_classes=256)

# TODO: make it parse from text file
def datagen(batch_size):
    
    first_byte = []
    second_byte = []
    operation = []
    target = []
    for batch in range(batch_size):
        a = "".join([hex_vocab[random.randint(0, 15)] for i in range(16)])
        b = "".join([hex_vocab[random.randint(0, 15)] for i in range(16)])
        op = random.choice(list(ops.keys()))
        s = hex(ops[op](int(a, 16), int(b, 16)))
        # print(s)
        s = format_targ(s)
        # if('x' in s):
        #     print("\t\t", s)

        first_byte.append(byte_tokenize(a))
        second_byte.append(byte_tokenize(b))
        operation.append(op_tokenize(op))
        target.append(one_hot(byte_tokenize(s)).flatten(start_dim=0, end_dim=1))

    first_byte = torch.cat(first_byte).view(batch_size, 8)
    second_byte = torch.cat(second_byte).view(batch_size, 8)
    bytes_ = torch.cat([first_byte, second_byte], dim=-1).view(batch_size, 2, 8)
    operation = torch.tensor(operation).view(batch_size, 1)
    target = torch.cat(target).view(batch_size, 256*8)

    return (bytes_, operation), target.float()