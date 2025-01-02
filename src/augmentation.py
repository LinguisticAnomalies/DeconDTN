import numpy as np
from copy import deepcopy
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from math import ceil



def reverseTranslationDF(src, tgt, txt_col, df_in, save=False, outfile=None, device="cuda", max_length=512):
    df = deepcopy(df_in)

    #== Setup
    # forward translate
    mname_forward = 'Helsinki-NLP/opus-mt-{}-{}'.format(src, tgt)

    model_forward = MarianMTModel.from_pretrained(mname_forward).to(device)

    tokenizer_forward = MarianTokenizer.from_pretrained(mname_forward)

    # reverse translate
    mname_reverse = 'Helsinki-NLP/opus-mt-{}-{}'.format(tgt, src)

    model_reverse = MarianMTModel.from_pretrained(mname_reverse).to(device)

    tokenizer_reverse = MarianTokenizer.from_pretrained(mname_reverse)

    #== Translate
    n_per_batch = 15
    translated_text_forward = []
    translated_text_reverse = []

    n_chunks = ceil(len(df)/n_per_batch)
    for tmp in tqdm(np.array_split(df, n_chunks), file=open("../log/reverseTranslate.txt", "w")):

        sample_text = list(tmp[txt_col])
        
        translated_forward = model_forward.generate(**tokenizer_forward(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device))
        _translated_text_forward = [tokenizer_forward.decode(t, skip_special_tokens=True) for t in translated_forward]


        translated_reverse = model_reverse.generate(**tokenizer_reverse(_translated_text_forward, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device))
        _translated_text_reverse = [tokenizer_reverse.decode(t, skip_special_tokens=True) for t in translated_reverse]

        
        translated_text_forward.extend(_translated_text_forward)
        translated_text_reverse.extend(_translated_text_reverse)


    df['text_translated_forward'] = translated_text_forward
    df['text_translated_reverse'] = translated_text_reverse

    if save:
        df.to_csv(outfile, index=False)

    df['Text'] = df['text_translated_reverse']
    df.drop(['text_translated_forward', 'text_translated_reverse'], axis=1, inplace=True)
    
    return df



