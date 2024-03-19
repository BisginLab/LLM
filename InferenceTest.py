from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel  # 0.5.0

# Load Models
base_model = "NousResearch/Llama-2-13b-hf"
peft_model = "FinGPT/fingpt-sentiment_llama2-13b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "cuda:0", load_in_8bit = True,)
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()

#input_ids = input_ids.to('cuda')

# Make prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .
Answer: ''',
]

# Generate results
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]

# show results
for sentiment in out_text:
    print(sentiment)

import pandas as pd
apple = pd.read_csv('./AAPL_news.csv')
prompt_new = pd.DataFrame({'input': apple['summary']})
#prompt_new['input'] = apple['content']
prompt_new['instruction'] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}"
prompt_new['output'] = ""
prompt_new.head()
print(prompt_new.shape)

prompt_list = []
for index, row in prompt_new.iterrows():
    if len(row['input'])>100:
        continue
    prompt = f'''Instruction: {row['instruction']}
Input: {row['input']}
Answer: {row['output']}'''
    prompt_list.append(prompt)

# Generate results
print(len(prompt_list))

'''
tokens = tokenizer(prompt_list, return_tensors='pt') 
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]
'''

# Tokenize prompts
tokens = tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt')

# Generate results
res = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], max_length=512)

# Decode the generated sequences
res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]

# Extract the answer part from each decoded sequence
out_text = [o.split("Answer: ")[1] for o in res_sentences]


# show results
for sentiment in out_text:
    print(sentiment)