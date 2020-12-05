import re


def remove_url(tweet):
        return (re.sub(r'http\S+', '', tweet))
    

def remove_non_alpha(tweet):
    return (re.sub(r'[^\x20-\x7E]', '', tweet))