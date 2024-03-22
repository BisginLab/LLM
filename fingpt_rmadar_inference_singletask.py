# Python Libraries that are required
# sentencepiece accelerate torch peft datasets bitsandbytes protobuf transformers==4.32.0 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Adjust these paths as needed
model_path = f"/home/rmadar/ondemand/data/sys/myjobs/projects/FinGPT/finetuned_model"
data_path = f"/home/rmadar/ondemand/data/sys/myjobs/projects/FinGPT/AAPL_news_cleaned.csv"

# Load the tokenizer and model
base_model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model.eval()
model.to("cuda:0")

# Load and prepare the data
apple_news = pd.read_csv(data_path)
prompt_new = pd.DataFrame({'input': apple_news['content']})
prompt_new['instruction'] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}"
prompt_new['output'] = ""

prompt_list = []
for index, row in prompt_new.iterrows():
    prompt = f'''Instruction: {row['instruction']}
Input: {row['input']}
Answer: {row['output']}'''
    prompt_list.append(prompt)

# Initialize a list to store outputs along with their prompts
output_with_prompts = []

for prompt in prompt_list:
    # Tokenize the prompt with truncation and a max_length that fits your model's capacity
    tokens = {k: v.to("cuda:0") for k, v in tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).items()}  # Adjust max_length as needed
    # Generate responses with a fixed max number of new tokens
    res = model.generate(**tokens, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)  # Adjust max_new_tokens as needed
    res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
    for o in res_sentences:
        sentiment = "Unknown"
        if "Answer: " in o:
            sentiment = o.split("Answer: ")[1].strip()
        output_with_prompts.append((sentiment, prompt))


# Show results with sentiment and the associated prompt
for sentiment, prompt in output_with_prompts:
    print(f"Sentiment: {sentiment}\nPrompt:\n{prompt}\n---")
