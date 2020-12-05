import re


def remove_url(tweet):
        return (re.sub(r'http\S+', '', tweet))
    

def remove_non_alpha(tweet):
    return (re.sub(r'[^\x20-\x7E]', '', tweet))


def pad_list(tokens, max_seq_len):
        while len(tokens) <= max_seq_len:
            tokens.append('0')
        return tokens
    
    
def trim_list(tokens, max_seq_len):
    return tokens[0:max_seq_len]