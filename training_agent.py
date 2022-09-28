import subprocess

import lda_training

num_topic_cf = (8, 100)
num_words_cf = (10, 100)
epoches_cf = [10, 50, 100, 500, 1000, 5000]
global_topic_cf = [True, False]


def get_command(num_topic, num_words, epoches, global_topic):
    cmds = ["python", "lda_training.py", f"--num_topic={num_topic}", f"--num_words={num_words}", f"--epoches={epoches}"]
    if global_topic:
        cmds.append("--global_topic=True")
    print(f'\n------\n{cmds}')
    return cmds


if __name__ == "__main__":
    # for num_topic in range(num_topic_cf[0], num_topic_cf[1]):
    #     for num_words in range(num_words_cf[0], num_words_cf[1]):
    #         for epoches in epoches_cf:
    #             for global_topic in global_topic_cf:
    #                 cmds = get_command(num_topic, num_words, epoches, global_topic)
    #                 subprocess.call(cmds)

    for global_topic in global_topic_cf:
        lda_training.train(num_topic_cf, num_words_cf, epoches_cf, global_topic)
