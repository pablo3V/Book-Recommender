###########################################################################################
#                                                                                         #
#                                 contains_all_genres                                     #
#                                                                                         #
# Function to check if all the specified genres in a given list are present in any        #
# of the genre columns of the DataFrame row.                                              #
#                                                                                         #
# The any function returns True if the current genre is found in any of the               #
#   7 genre columns.                                                                      #
# The all function ensures that this condition (any returning True) holds for             #
#   every genre in the genres list.                                                       #
#                                                                                         #
###########################################################################################

def contains_all_genres(row, genres):
    return all(any(row[f'Genre_{i}'] == genre for i in range(1, 8)) for genre in genres)
    
    
    
###########################################################################################
#                                                                                         #
#                                 contains_any_genre                                      #
#                                                                                         #
# Function to check if any the specified genres in a given list are present in any        #
# of the genre columns of the DataFrame row.                                              #
#                                                                                         #
# The any function returns True if at least one genre from the genres list is found       #
# in any of the 7 genre columns of the row.                                               #
#                                                                                         #
###########################################################################################

def contains_any_genre(row, genres):
    return any(row[f'Genre_{i}'] in genres for i in range(1, 8))
    
    
    
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
  
def books_satisfying_genres(data, genres, exclude=[], combine=False): 
    
    data = pd.merge(data, books_genres[['BookID','Genre_1', 'Genre_2', 'Genre_3', 'Genre_4', 'Genre_5', 'Genre_6', 'Genre_7']], on='BookID', how='left')

    if len(genres) == 0: # Case with no genres specified
        data = data
    else:
        if combine: # If the user wants the books to include all the genres specified
            if len(genres) <= 7:
                data = data[data.apply(lambda row: contains_all_genres(row, genres), axis=1)]
            else:
                print('Books can have, at most, 7 different genres. If you want book recommendations including all the selected genres simultaneously, please, choose a maximum of 7 options.')
                return
        else: # If the user wants the books to include at least one of the genres specified
            data = data[data.apply(lambda row: contains_any_genre(row, genres), axis=1)]

    # To drop the books with, at leat, one of its genres in the list exclude
    data = data[~data.apply(lambda row: contains_any_genre(row, exclude), axis=1)]

    return data # This dataframe contains the genres of the books 
    
    
    
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
