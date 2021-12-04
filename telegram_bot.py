from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

with open('token.txt', 'r') as token:
    TOKEN = token.read()
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

import torch
import transformers
import pandas as pd
import argparse
import os
import numpy as np

import dataset
import transform 
import encoder

def token_proc(token, st):
    if st and token[0]=='[' and token[-1]==']':
        token=''
    elif token[:2]!='##':
        token = ' '+token
    else:
        token = token[2:]
    return token
def convert_ids_to_str(ids, tokenizer, st=False):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    #print(tokens)
    tokens = list(map(lambda x:token_proc(x, st), tokens))
    return ''.join(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--model_name", default='pytorch_model.bin', type=str)
    parser.add_argument("--architecture", required=True, type=str, help='[poly, bi, cross]')
    parser.add_argument("--poly_m", default=0, type=int, help="Number of m of polyencoder")
    parser.add_argument("--max_contexts_length", default=128, type=int)
    parser.add_argument("--max_response_length", default=32, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)
    parser.add_argument("--out_base", type=str, help="Path to computed cadidate base.")
    args = parser.parse_args()
    print(args)
    
    ## init dataset and bert model
    tokenizer = transformers.BertTokenizerFast.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True, clean_text=False, truncation=True)
    context_transform = transform.SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
    response_transform = transform.SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
    concat_transform = transform.SelectionConcatTransform(tokenizer=tokenizer, max_len=args.max_response_length+args.max_contexts_length)
    
    train_dataset = dataset.SelectionDataset(os.path.join(args.train_dir, 'train.txt'),
                                                                  context_transform, response_transform, concat_transform, sample_cnt=None, mode=args.architecture)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = transformers.BertConfig.from_json_file(os.path.join(args.bert_model, 'config.json'))
    bert_config.vocab_size = 119548
    previous_model_file = os.path.join(args.bert_model, args.model_name) 
    print('Loading parameters from', previous_model_file)
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    bert = transformers.BertModel.from_pretrained(args.bert_model, state_dict=model_state_dict)
    del model_state_dict
    if args.architecture == 'poly':
        model = encoder.PolyEncoder(bert_config, bert=bert, poly_m=args.poly_m, tokenizer=tokenizer)
    elif args.architecture == 'bi':
        model = encoder.BiEncoder(bert_config, bert=bert)
    elif args.architecture == 'cross':
        model = encoder.CrossEncoder(bert_config, bert=bert)
    else:
        raise Exception('Unknown architecture.')
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)
    model.eval()
    ########################
    
            
            
dialog_history = []    
@dp.message_handler(content_types=['text'])
async def get_text_messages(msg: types.Message):
    inp_string = msg.text
    if msg.text == '!end':
        global dialog_history
        dialog_history = []  
    dialog_history.append(inp_string)
    ########################
    batch = context_transform(dialog_history)
    batch = tuple(torch.tensor([t]).to(device) for t in batch)
    if args.architecture == 'cross':
        raise Exception('not implemented yet')
        text_token_ids_list_batch, text_input_masks_list_batch = batch
        loss = model(text_token_ids_list_batch, text_input_masks_list_batch)
    else:
        context_token_ids_list_batch, context_input_masks_list_batch = batch
        out = model(context_token_ids_list_batch, context_input_masks_list_batch).cpu().detach().numpy()
        print(out.shape)
        relevant_response = [None]
        relevant_sim = [0]
        with open(args.out_base, 'r') as base:

            for step, i in enumerate(base.readlines()):
                ids, embd = i.split('|||')
                ids = np.array([float(i) for i in ids.split(' ')])
                embd = np.array([float(i) for i in embd.split(' ')])
                cos_sim = np.matmul(out, embd)
                if cos_sim > relevant_sim[-1]:
                    relevant_sim.append(cos_sim)
                    relevant_response.append(ids)
                if len(relevant_response)>10:
                    relevant_response = relevant_response[1:]
                    relevant_sim = relevant_sim[1:]
        responce = convert_ids_to_str(relevant_response[-1], tokenizer, True)
        dialog_history.append(responce)
    await msg.answer(ыек(dialog_history)+'\n'+responce)
      
if __name__ == '__main__':
   executor.start_polling(dp)
