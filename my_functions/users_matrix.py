import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors



########################################################################################
#                                                                                      #
#                                user_dictionary_to_df                                 #
#                                                                                      #
# Function to convert the new user dictionary of ratings into a dataframe.             #
#                                                                                      #
########################################################################################

def user_dictionary_to_df(user_dict, target_user, books, ratings):
    # Convert the target_user dictionary to a dataframe with the required columns
    target_user_ratings = pd.DataFrame(user_dict.items(), columns=['Title', 'Rating'])
    target_user_ratings = pd.merge(target_user_ratings, books[['Title', 'BookID']], on='Title', how='left').drop('Title', axis=1)

    # Add an ID to the target_user
    target_user_ratings['UserID'] = target_user 

    # Concat the target_user ratings to the original ratings dataframe
    data = pd.concat([ratings, target_user_ratings], ignore_index=True)

    return data



########################################################################################
#                                                                                      #
#                                  selected_users_df                                   #
#                                                                                      #
# Function to get the users that have, at least, one common book rated with the        #
# target user. There is a maximum number of users to keep to ensure that we do not     #
# run out of memory.                                                                   #
#                                                                                      #
########################################################################################

def selected_users_df(ratings_df, target_books, n_max, target_user):
    # Users who have rated at least 1 of the items rated by the current user
    selected_users = ratings_df[ratings_df['BookID'].isin(target_books)]
    selected_users = pd.DataFrame(selected_users.groupby('UserID').size(), columns=['Coincidences']).sort_values(by='Coincidences', ascending=False).reset_index()

    number_of_users = selected_users.shape[0]

    # Drop users if the size of selected_users is too large
    if number_of_users > n_max:
        # First, select the first n_users_upper_limit with higher coincidences
        selected_users_aux = selected_users.head(n_max) 
        # Minimum number of coincidences in the previous set
        lowest_coincidences = selected_users_aux['Coincidences'].min() 
        # Number of users with the minimum number of coincidences needed to reach n_users_upper_limit
        n_users_with_lowest_coincidences_needed = n_max - selected_users[selected_users['Coincidences'] > lowest_coincidences].shape[0]

        selected_users_higher = selected_users[selected_users['Coincidences'] > lowest_coincidences]
        selected_users_lowest = selected_users[selected_users['Coincidences'] == lowest_coincidences].sample(n_users_with_lowest_coincidences_needed)

        # Users with the highest number of coincidences and random users with the minimum number of coincidences
        selected_users = pd.concat([selected_users_higher, selected_users_lowest], ignore_index=True)

    if len(selected_users[selected_users['UserID'] == target_user]) == 0:
            target_user = pd.DataFrame({target_user: len(target_books)}.items(), columns=['UserID', 'Coincidences'])
            selected_users = pd.concat([selected_users, target_user], ignore_index=True)
        
    # Ratings of the selected users
    selected_ratings = ratings_df[ratings_df['UserID'].isin(selected_users.UserID.values)]

    return selected_users, selected_ratings



########################################################################################
#                                                                                      #
#                                  get_users_matrix                                    #
#                                                                                      #
# Function to create the matrix with the selected users and the books they have rated. #
# It returns the same matrix both in csr and dataframe formats.                        #
#                                                                                      #
########################################################################################


# The new version of the function takes into account only the books that the target user
# has rated. 
# In our opinion, what is really important is that the users have coincidences in the books 
# the target user has rated, and we do not care about the other books. For example:
# 
# target user has rated N books with some ratings.
# user_1 has rated the same N books with the same ratings, plus 1 additional book
# user_2 has rated the same N books with the same ratings, plus 1000 additional books
#
# The user_1 will be way closer to the target_user, according to the KNN algorithm,
# than user_2, despite having the same ratings on the important N books. This is because the 
# dimensionality of the space matters. 
#
# Then, we think it is better to consider the target_books, not only because computationally is
# highly cheaper, but also because it makes more sense to us.

def get_users_matrix(ratings, target_books):
    # Creating the matrix with users and books
    ratings = ratings[ratings['BookID'].isin(target_books)]
    ratings_matrix = ratings[['UserID', 'BookID', 'Rating']].pivot(index = 'UserID', columns = 'BookID', values = 'Rating').fillna(0)
    ratings_csr_matrix = csr_matrix(ratings_matrix.values)
    # Normalize the csr matrix
    ratings_csr_matrix_norm = get_csr_matrix_norm(ratings_csr_matrix, method='min_max')

    users = list(ratings_matrix.index)
    
    return ratings_csr_matrix_norm, users
  
    
def get_users_matrix_previous_version(ratings):
    # Creating the matrix with users and books
    ratings_matrix = ratings[['UserID', 'BookID', 'Rating']].pivot(index = 'UserID', columns = 'BookID', values = 'Rating').fillna(0)
    ratings_csr_matrix = csr_matrix(ratings_matrix.values)
    # Normalize the csr matrix
    ratings_csr_matrix_norm = get_csr_matrix_norm(ratings_csr_matrix, method='min_max')

    users = list(ratings_matrix.index)
    
    return ratings_csr_matrix_norm, users



########################################################################################
#                                                                                      #
#                                      knn_model                                       #
#                                                                                      #
# Function to get the potential recommendations to the target user.                    #
# It makes use of a KNN algorithm and a weighted rating get the main recommendation.   #
#                                                                                      #
########################################################################################



def knn_model(csr_matrix, users, target_user, number_neighbours, ratings, exclude):
    # Build the KNN model
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(csr_matrix)

    # Get the index of the target_user
    target_user_index = users.index(target_user)

    if len(users) < (number_neighbours + 1):
        number_neighbours = len(users)

    # Look for the closest neighbours of the target_user
    distances, indices = model_knn.kneighbors(csr_matrix[target_user_index], n_neighbors=(number_neighbours + 1))

    # Most similar users
    most_similar_users_indices = indices.flatten()[1:]

    # Get the UserIDs corresponding to the numerical indices
    most_similar_users = [users[i] for i in most_similar_users_indices]

    # Ratings of the most similar users
    similar_users_ratings = ratings[(ratings['UserID'].isin(most_similar_users)) & (~ratings['BookID'].isin(exclude))]

    # Weights from the distances for the weighted average
    alpha = 5.0
    weights = [np.exp(- alpha * distance) for distance in distances.flatten()[1:]]
    # Create a DataFrame for the weights
    weights_df = pd.DataFrame({'UserID': most_similar_users, 'Weight': weights})
    
    
    #print('Distances: ', distances.flatten()[1:])
    #print('Weights: ', weights)
    
    
    # Merge the weights with the ratings
    weighted_ratings = similar_users_ratings.merge(weights_df, on='UserID')
    # Calculate the weighted rating for each book
    weighted_ratings['Weighted_Rating'] = weighted_ratings['Rating'] * weighted_ratings['Weight']

    # Aggregate weighted ratings by book
    book_weighted_avg = weighted_ratings.groupby('BookID').agg(
        Weighted_Sum=('Weighted_Rating', 'sum'),
        Weight_Sum=('Weight', 'sum'),
        Ratings_Count=('Rating', 'count')
    ).reset_index()

    # Calculate the weighted average rating
    book_weighted_avg['Weighted_Average_Rating'] = book_weighted_avg['Weighted_Sum'] / book_weighted_avg['Weight_Sum']
    similar_books = book_weighted_avg.copy()

    # In order to obtain a weighted rating for each book taking into account the number of votes a book has and its average rating, 
    # I will use the "True Bayesian Estimate", used by IMDB 
    # (https://stats.stackexchange.com/questions/6418/rating-system-taking-account-of-number-of-votes, 
    #  https://github.com/wbotelhos/rating):

    # Weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C 

    # , where:

    # - R = average for the book (mean)
    # - v = number of votes for the book
    # - m = minimum votes required to be listed in the top
    # - C = the mean vote across the whole set of books

    # For an intuitive explanation of the formula, take a look to [this link](https://stats.stackexchange.com/questions/189658/what-are-good-resources-on-bayesian-rating).

    # WR = P(enough evidence) × (Rating based on evidence) + P(no evidence) × (best guess when no evidence)
    
    # Parameters for the "True Bayesian Estimate". We can chose several options for m.
    R = similar_books['Weighted_Average_Rating']
    v = similar_books['Ratings_Count']
    #m = similar_books['Ratings_Count'].mean()
    m = similar_books['Ratings_Count'].quantile(0.9)
    C = weighted_ratings['Weighted_Rating'].sum() / weighted_ratings['Weight'].sum()
    # C is the mean of all the ratings, not the mean of the books' means

    # Weighted Rating: "True Bayesian Estimate"
    similar_books['Weighted_Rating'] = (v / (v + m)) * R + (m / (v + m)) * C

    similar_books = similar_books.sort_values(by='Weighted_Rating', ascending=False).reset_index()
    
    return similar_books



########################################################################################
#                                                                                      #
#                                 get_csr_matrix_norm                                  #
#                                                                                      #
# Function to normalize the csr matrix with the ratings and the users.                 #
# There are three different methods for the normalization.                             #
#                                                                                      #
########################################################################################

def get_csr_matrix_norm(csr_matrix, method='mean_centering'):
    csr_matrix_norm = csr_matrix.copy()

    for i in range(csr_matrix.shape[0]):
        row_start = csr_matrix.indptr[i]
        row_end = csr_matrix.indptr[i + 1]

        if row_start < row_end: # If the row is not empty
            row_data = csr_matrix_norm.data[row_start:row_end]
    
            if method == 'mean_centering':
                # Normalize each row subtracting by the mean value of the row
                mean = row_data.mean()
                row_data = row_data - mean 
            elif method == 'z_score':
                # Normalize each row subtracting the mean and dividing by the std of the row
                mean = row_data.mean()
                std = row_data.std()
                if std != 0: 
                    row_data = (row_data - mean) / std
                else:
                    row_data = row_data - mean
            elif method == 'min_max': 
                # Normalize each row subtracting the min of the row and dividing by the difference between max and min of each row
                max_val = row_data.max()
                min_val = row_data.min()
                if max_val != min_val:
                    row_data = (row_data - min_val) / (max_val - min_val) 
                else:
                    row_data = row_data - min_val   
            else:
                raise ValueError("Invalid method. Please specify 'mean_centering', 'z_score', or 'min_max'.")

            csr_matrix_norm.data[row_start:row_end] = row_data
            
    return csr_matrix_norm
