from collections import Counter

vocab_size = 10_000
sets_of_words = []
for i in range(1, 6):
    words = ' '.join(open(f'/home/student/Code/data/new/sentiment.train.{i}', 'r')).replace('\n', ' ').split(' ')
    c = Counter()
    c.update(words)
    sets_of_words.append([tok for tok, _ in c.most_common(10_000) if tok != ' '])

final_vocab = set()
for i in range(10_000):
    for j in range(5):
        if i > len(sets_of_words[j]):
            continue
        final_vocab.add(sets_of_words[j][i])
    if len(final_vocab) >= 10_000:
        break
open('/home/student/Code/data/new/vocab.txt', 'w').writelines('\n'.join(list(final_vocab)))

