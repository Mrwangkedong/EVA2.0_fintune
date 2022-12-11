from model import EVAModel, EVATokenizer

tokenizer = EVATokenizer.from_pretrained("/opt/data/private/nlp03/kdwang/huggingface_models/EVA2-base")
model = EVAModel.from_pretrained("/opt/data/private/nlp03/kdwang/huggingface_models/EVA2-base")
model = model.half().cuda()


# def gen(input_str):
#     tokenize_out = tokenizer(input_str, "", return_tensors="pt", padding=True, truncation=True, max_length=32)
#     input_ids = tokenize_out.input_ids.cuda()
#     generation = model.generate(input_ids, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9,
#                           max_length=32, use_cache=True)
#     return tokenizer.decode(generation[0], skip_special_tokens=True)


input_str = "你好，<mask>"

tokenize_out = tokenizer(input_str, "", return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenize_out.input_ids.cuda()
print(tokenize_out)

generation = model.generate(input_ids, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9,
                            max_length=32, use_cache=True)
print(tokenizer.decode(generation[0], skip_special_tokens=True))
