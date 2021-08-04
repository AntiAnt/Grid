import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("adj")
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    
    rate = args.lr + 1.0
    print(args)
    print(f'This is the {args.adj} script')
    print(f'Learning rate + 1 = {rate}')