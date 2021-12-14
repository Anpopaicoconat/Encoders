def get_dialog(inp, mod):
    p1 = inp['persona_1_profile'].replace('<span class=participant_2>', '').replace('<span class=participant_1>', '').replace('</span>', '').replace('\r', '').replace('\n', '').split('<br />')[:-1]
    p2 = inp['persona_2_profile'].replace('<span class=participant_2>', '').replace('<span class=participant_1>', '').replace('</span>', '').replace('\r', '').replace('\n', '').split('<br />')[:-1]
    #print(len(tokenizer('\n'.join(p1))['input_ids']))
    inp = inp['dialogue'].replace('<span class=participant_2>', '').replace('<span class=participant_1>', '').replace('</span>', '').replace('\r', '').replace('\n', '').split('<br />')
    if inp[0][15:] == 'Пользователь 1: ':
        out = [[inp[0], p1]]
    else:
        out = [[inp[0], p2]]

    for i in inp[1:]:
        if i[:15]==out[-1][0][:15] and mod == 'join':
            out[-1][0] = out[-1][0] + ' ' + i[15:]
        elif i[:12] != 'Пользователь' and mod == 'join':
            out[-1][0] = out[-1][0] + ' ' + i
        else:
            if i[:15] == 'Пользователь 1:':
                out.append([i, p1])
            else:
                out.append([i, p2])
    out = [[i[0].replace('Пользователь 1: ', '').replace('Пользователь 2: ', ''),i[1]] for i in out]
    #print(out)
    return out

def proc_row(row, data, negs_n, out_file, mod):
    ind = row[0]
    row = row[1]
    dialog = get_dialog(row, mod=mod)
    with open(out_file, 'a') as outfile:
        for i in range(1, len(dialog)):
            context = dialog[:i]
            res = dialog[i]
            negs = dialog[i+1:][:negs_n]
            while len(negs) != negs_n:
                samp = data.sample().iloc[0]
                if samp.name != ind:
                    samp = np.random.choice([i[0] for i in get_dialog(samp, mod=mod)])
                    negs.append(samp)
            context = [i[0] for i in context]
            #print('>', context)
            #print(res[0], res[1])
            outfile.write('{}\t{}\t{}\t{}\n'.format(1, '\t'.join(context), '\t'.join(res[1]), res[0].strip()))
            for neg in negs:
                outfile.write('{}\t{}\t{}\t{}\n'.format(0, '\t'.join(context), '\t'.join(res[1]), neg[0].strip()))
        
    

def get_datasets(df, split_i, negs_n, mod, path='train.txt'):
    data = df.sample(len(df))
    if split_i != 0:
        train_path='{}/train.txt'.format(path)
        val_path='{}/dev.txt'.format(path)
        split_i = len(data)//split_i
        train = data[split_i:]
        val = data[:split_i]
    else:
        train_path='{}/split.txt'.format(path)
        val_path='{}/val.txt'.format(path)
        train = data
        val = data[:1]
    with open(train_path, 'w') as f:
        pass
    with open(val_path, 'w') as f:
        pass
    for row in train.iterrows():
        proc_row(row,train, negs_n, train_path, mod=mod)
    print(len(train))
    for row in val.iterrows():
        proc_row(row,train, negs_n, val_path, mod=mod)
    print(len(val))
