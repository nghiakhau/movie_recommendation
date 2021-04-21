from torch.utils.data import Dataset
from utils.utils import genre2label


class MovieDataset(Dataset):
    def __init__(self, movie_dataframe):
        super().__init__()

        self.data_column = "text"
        self.class_column = "label"

        # Create label from genres
        genres_list = [x for sublist in movie_dataframe.genres.tolist() for x in sublist]
        genres = set(genres_list)
        genre2id = dict((pid, i) for (i, pid) in enumerate(genres))
        num_genre = len(genres)
        movie_dataframe[self.class_column] = movie_dataframe['genres'].map(
                                                lambda genre: genre2label(genre, genre2id, num_genre))

        self.data = movie_dataframe[[self.data_column, self.class_column]]

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], self.data.loc[idx, self.class_column]

    def __len__(self):
        return self.data.shape[0]




