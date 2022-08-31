import argparse
import configparser


def write2file(file_path: str, contents, mode='w'):
    if file_path is None:
        raise ValueError("[REQUIRED] file_path is None!")

    with open(file_path, mode=mode) as fi:
        # Writing data to a file
        fi.write(str(contents))


def read_ini(file_path, section='APP'):
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        for key in config[section]:
            print((key, config[section][key]))
    return config[section]


def get_configs(init_file=None):
    if init_file:
        config = read_ini('config.ini')
        # print(f"Config: {config}")

        # Configurate variables
        num_topics = int(config['NUM_TOPIC'])
        num_words = int(config['NUM_WORDS'])
        epoches = int(config['EPOCHES'])
        global_topic = bool(config['GLOBAL_TOPIC'])
        # return [num_topics, num_words, epoches]
        return {'num_topics': num_topics, 'num_words': num_words, 'epoches': epoches, 'global_topic': global_topic}
    else:
        # Create the parser
        parser = argparse.ArgumentParser()
        # Add an argument
        parser.add_argument('--num_topic', type=int, required=True)
        parser.add_argument('--num_words', type=int, required=False, default=10)
        parser.add_argument('--epoches', type=int, required=True)
        parser.add_argument('--global_topic', type=bool, required=False, default=False,
                            help='If global_topic = True, it only train LDA for global. \
                            If not, it trains for each grouped hotel.')
        # Parse the argument
        args = parser.parse_args()
        # Print "Hello" + the user input argument
        # return [args.num_topic, args.num_words, args.epoches]
        return {'num_topics': args.num_topic, 'num_words': args.num_words, 'epoches': args.epoches,
                'global_topic': args.global_topic}
