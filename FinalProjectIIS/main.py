import pandas as pd
from tqdm import tqdm

PATH = '/home/student/yelp_academic_dataset_review.json'


def main():
    for chunk in tqdm(pd.read_json(PATH, chunksize=10_000, lines=True)):
        chunk
        pass
    pass


if __name__ == '__main__':
    main()
