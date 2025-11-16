from engine import Engine

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=('vast', 'pstance', 'covid'), default='vast',
                        help='which dataset to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=5e-5)
    parser.add_argument('--max_grad', type=float, default=0)
    parser.add_argument('--n_layers_freeze', type=int, default=0)
    parser.add_argument('--shared_model', type=str, choices=('bert-base', 'bertweet', 'covid-twitter-bert'),
                        default='bert-base')
    parser.add_argument('--wiki_model', type=str, choices=('', 'bert-base'), default='')
    parser.add_argument('--tweet_model', type=str, choices=('', 'bertweet'), default='')
    parser.add_argument('--google_model', type=str, choices=('', 'bert-base'), default='')
    parser.add_argument('--n_layers_freeze_wiki', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--inference', type=int, default=0, help='if doing inference or not')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--moe_top_k', type=int, default=2,)
    parser.add_argument('--enable_wiki', type=int, default=1,)
    parser.add_argument('--enable_google', type=int, default=1,)
    parser.add_argument('--enable_tweet', type=int, default=1,)
    args = parser.parse_args()
    print(args)

    engine = Engine(args)
    engine.train()