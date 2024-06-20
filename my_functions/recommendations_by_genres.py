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
    
    
    
###########################################################################################
#                                                                                         #
#                                  update_recommendations                                 #
#                                                                                         #
# Function to clean the potential recommendations.                                        #
# If there is a recommendation of a volume of a saga, we check if the user has read any   #
# volume. If so, we recommend the volume that follows inmediatedly the one that the user  #
# has last read. If not, we recommend the first volume.                                   #
#                                                                                         #
########################################################################################### 

def update_recommendations(recommendations, user_books, books):
            
    # Get the list of read sagas by the target user with the target user
    user_sagas = user_books.dropna(subset=['Saga_Name'])

    # Create a diccionary with the latest volume read of a saga
    user_saga_volumes = user_sagas.groupby('Saga_Name')['Saga_Volume'].max().to_dict()

    updated_recommendations = {'BookID': []}

    for index, row in recommendations.iterrows():
        saga_name = row['Saga_Name']
        saga_volume = row['Saga_Volume']
        bookid = row['BookID']

        # If the book is not part of a saga, it is kept in the list of recommendations
        if pd.isna(saga_name): # if saga_name = nan
            updated_recommendations['BookID'].append(bookid)
            continue
            
        # If the book is part of a saga, we verify the last volume read by the target_user
        if saga_name in user_saga_volumes:
            last_read_volume = user_saga_volumes[saga_name]

            if saga_volume == last_read_volume + 1:
                # If the volume is the following in the saga, the book is recommended
                updated_recommendations['BookID'].append(bookid)

            else:
                # If not, the next volume is recommended
                next_volume = last_read_volume + 1
                next_volume = books[(books['Saga_Name'] == saga_name) & 
                                    (books['Saga_Volume'] == next_volume)]['BookID'].values
                if len(next_volume) > 0:
                        updated_recommendations['BookID'].append(next_volume[0])

        else:
            # If the user has not read any volume of the saga, the first volume is recommended
            first_volume = books[(books['Saga_Name'] == saga_name) & 
                                 (books['Saga_Volume'] == 1)]['BookID'].values
            if len(first_volume) > 0:
                updated_recommendations['BookID'].append(first_volume[0])

    # Create a new dataframe with the updated recommendations and the duplicates dropp
    updated_recommendations_df = pd.DataFrame(updated_recommendations).drop_duplicates(subset=['BookID']).reset_index(drop=True)
    
    return updated_recommendations_df
    
