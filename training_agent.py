import subprocess

num_topic_cf = (8, 50)
num_words_cf = (8, 60)
epoches_cf = [100, 1000, 5000, 10000, 50000, 100000]
global_topic_cf = [True, False]


def get_command(num_topic, num_words, epoches, global_topic):
    cmds = ["python", "lda_training.py", f"--num_topic={num_topic}", f"--num_words={num_words}", f"--epoches={epoches}"]
    if global_topic:
        cmds.append("--global_topic=True")
    print(f'\n------\n{cmds}')
    return cmds


if __name__ == "__main__":
    for num_topic in range(num_topic_cf[0], num_topic_cf[1]):
        for num_words in range(num_words_cf[0], num_words_cf[1]):
            for epoches in epoches_cf:
                for global_topic in global_topic_cf:
                    cmds = get_command(num_topic, num_words, epoches, global_topic)
                    subprocess.call(cmds)
