import os
import socket

if __name__ == '__main__':
    data = 'vast'
    batch_size = 4  # Keep small for MoE memory efficiency
    epochs = 50
    patience = 2
    lr = 1e-5  # Lower lr for MoE stability
    l2_reg = 1e-4  # Increased regularization for MoE
    shared_model = 'bert-base'
    wiki_model = 'bert-base'
    google_model = 'bert-base'
    tweet_model = 'bertweet'
    n_layers_freeze = 0
    n_layers_freeze_wiki = 0
    gpu = '1'
    inference = 0
    moe_top_k = 1

    if wiki_model == shared_model:
        n_layers_freeze_wiki = n_layers_freeze
    if not wiki_model or wiki_model == shared_model:
        n_layers_freeze_wiki = 0

    os.makedirs('results', exist_ok=True)
    if data != 'vast':
        file_name = f'results/{data}-lr={lr}-bs={batch_size}.txt'
    else:
        file_name = f'results/{data}-moe-lr={lr}-bs={batch_size}.txt'

    if shared_model != 'bert-base':
        file_name = file_name[:-4] + f'-{shared_model}.txt'
    if n_layers_freeze > 0:
        file_name = file_name[:-4] + f'-n_layers_fz={n_layers_freeze}.txt'
    if wiki_model:
        file_name = file_name[:-4] + f'-wiki={wiki_model}.txt'
    if n_layers_freeze_wiki > 0:
        file_name = file_name[:-4] + f'-n_layers_fz_wiki={n_layers_freeze_wiki}.txt'

    n_gpus = len(gpu.split(','))
    file_name = file_name[:-4] + f'-n_gpus={n_gpus}.txt'

    command = f"python3 -u src/train.py " \
              f"--data={data} " \
              f"--shared_model={shared_model} " \
              f"--wiki_model={wiki_model} " \
              f"--tweet_model={tweet_model} " \
              f"--google_model={google_model} " \
              f"--n_layers_freeze={n_layers_freeze} " \
              f"--n_layers_freeze_wiki={n_layers_freeze_wiki} " \
              f"--batch_size={batch_size} " \
              f"--epochs={epochs} " \
              f"--patience={patience} " \
              f"--lr={lr} " \
              f"--l2_reg={l2_reg} " \
              f"--gpu={gpu} " \
              f"--inference={inference} " \
              f"--moe_top_k={moe_top_k}"
              # f" > {file_name}"

    print(command)
    os.system(command)