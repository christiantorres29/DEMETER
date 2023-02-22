import argparse

# other code. . .

def main(name):
    print('Hello, %s!' % name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='your name, enter it')
    args = parser.parse_args()

    main(args.name)