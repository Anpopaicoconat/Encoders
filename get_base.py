import torch
import transformers
import pandas as pd
import argparse
import os
import csv
import numpy as np
import faiss

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
    parser.add_argument("--train_dir", default='', type=str)
    parser.add_argument("--base_name", default='train', type=str, help="название датасета используемого для получения базы")
    parser.add_argument("--out_base", type=str, help="Path to computed cadidate base.")
    parser.add_argument("--faiss", type=bool, help="Use faiss")
    args = parser.parse_args()
    print(args)
    
    ## init dataset and bert model
    tokenizer = transformers.BertTokenizerFast.from_pretrained(os.path.join(args.bert_model, "vocab.txt"), do_lower_case=True, clean_text=False, truncation=True)
    context_transform = transform.SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
    response_transform = transform.SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
    concat_transform = transform.SelectionConcatTransform(tokenizer=tokenizer, max_len=args.max_response_length+args.max_contexts_length)
    
    train_dataset = dataset.SelectionDataset(os.path.join(args.train_dir, args.base_name+'.txt'), context_transform, response_transform, concat_transform, sample_cnt=None, mode=args.architecture)
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
    
    if args.faiss:
        print(bert_config.pooler_fc_size)
        index = faiss.index_factory(bert_config.pooler_fc_size, ',IVF1080, Flat', faiss.METRIC_L2)
        list_for_fais = []
    else:
        base = open(args.out_base, 'w')
        
    L = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch[2:4])

        if args.architecture == 'poly':
            candidates_token_ids_list_batch, candidates_input_masks_list_batch = batch
            out = model(responses_input_ids=candidates_token_ids_list_batch, responses_input_masks=candidates_input_masks_list_batch, mod='get_base').cpu().detach().tolist()

            candidates_token_ids_list_batch = candidates_token_ids_list_batch.cpu().detach().tolist()
            for ids, embd in zip(candidates_token_ids_list_batch, out):
                ids=ids[0]
                string = '{}|||{}\n'.format(' '.join([str(i) for i in ids]), ' '.join([str(i) for i in embd]))
                base.write(string)

        elif args.architecture == 'bi':
            candidates_token_ids_list_batch, candidates_input_masks_list_batch = batch
            out = model(responses_input_ids=candidates_token_ids_list_batch, responses_input_masks=candidates_input_masks_list_batch, mod='get_base').cpu().detach().numpy()
            context_token_ids_list_batch = context_token_ids_list_batch.cpu().detach().numpy()
            
            if args.faiss:
                #index.add_with_ids(out, context_token_ids_list_batch)
                if list_for_faiss:
                    list_for_faiss = np.concatinate((list_for_faiss, out), axis=0)
                else:
                    list_for_faiss = out
            else:
                for ids, embd in zip(context_token_ids_list_batch, out):
                    string = '{}|||{}\n'.format(' '.join([str(i) for i in ids]), ' '.join([str(i) for i in embd]))
                    base.write(string)

        else:
            raise Exception('not implemented yet for this architecture')

        if step%10==0:
            print(step, L)
    if args.faiss:
        print()
        index.train(list_for_faiss)
        index.add(list_for_faiss)
        faiss.write_index(index, "flat.index")
    else: 
        base.close()
    
    
