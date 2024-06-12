import pandas as pd
import numpy as np
    
    
###########################################################################################
#                                                                                         #
#                              books_satisfying_genres                                    #
#                                                                                         #
# This function returns the books (with their genres) satisfying some genres restrictions #
#                                                                                         #
# data = dataset with the prediction                                                      #
# genres = list of the genres the user is interested in                                   #
# exclude = list of genres the user wants to exclude                                      #
# combine = if True, look for books that have all the genres                              #
#                                                                                         #
###########################################################################################

def books_satisfying_genres(data, books_genres, genres, exclude=[], combine=False): 
    
    merged_data = pd.merge(data, books_genres[['BookID', 'Genres', 'Genre_1', 'Genre_2', 'Genre_3', 'Genre_4', 'Genre_5', 'Genre_6', 'Genre_7']], on='BookID', how='left')

    if genres: # Case with any genre specified
        if combine: # If the user wants the books to include all the genres specified
            if len(genres) <= 7:
                genres_set = set(genres)
                mask = [genres_set.issubset(set(row)) for row in merged_data['Genres']]
                merged_data = merged_data[mask]
            else:
                print('Books can have, at most, 7 different genres. If you want book recommendations including all the selected genres simultaneously, please, choose a maximum of 7 options.')
                return
        else:
            # Check if any genre is present in any 'Genres' column
            genre_mask = np.logical_or.reduce([merged_data[f'Genre_{i}'].isin(genres) for i in range(1, 8)])
            merged_data = merged_data[genre_mask]

    # To drop the books with, at leat, one of its genres in the list exclude
    if exclude:
        # Exclude rows with any excluded genre
        exclude_mask = np.logical_not(np.logical_or.reduce([merged_data[f'Genre_{i}'].isin(exclude) for i in range(1, 8)]))
        merged_data = merged_data[exclude_mask]

    return merged_data # This dataframe contains the genres of the books 
    
    
    
###########################################################################################
#                                                                                         #
#                                      ordinal_number                                     #
#                                                                                         #
# Function that returns the ordinal equivalent of a number.                               #
#                                                                                         #
########################################################################################### 
    
def ordinal_number(number):
    if 10 <= number % 100 <= 20: # n % 100 returns the last 2 digits
        sufix = 'th'
    else:
        sufix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return str(number) + sufix
