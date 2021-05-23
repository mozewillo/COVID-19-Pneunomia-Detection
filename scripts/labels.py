import re
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor


def tokenizer(loaded_data, length=160):
    encoder = StaticTokenizerEncoder(
        loaded_data, 
        tokenize=lambda s: re.findall(r"[^\W\d_]+|\d+", s)
    )
    encoded_data = [encoder.encode(example) for example in loaded_data]
    encoded_data = [pad_tensor(x, length=length) for x in encoded_data]
    return stack_and_pad_tensors(encoded_data)
 
