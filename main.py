#############################################################################
#                                                                           #
#            DASH APPLICATION FOR THE BOOK RECOMMENDATION SYSTEM            #
#                                                                           #
#############################################################################

#######################################################################################################

# Libraries

import pandas as pd
import numpy as np
import ast
import itertools

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from io import StringIO

import os

import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_grocery

import json

from google.cloud import storage

#######################################################################################################

# Functions

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
# In my opinion, what is really important is that the users have coincidences in the books 
# the target user has rated, and I do not care about the other books. For example:
# 
# target user has rated N books with some ratings.
# user_1 has rated the same N books with the same ratings, plus 1 additional book
# user_2 has rated the same N books with the same ratings, plus 1000 additional books
#
# Obviously, the user_1 will be way closer to the target_user, according to the KNN algorithm,
# than user_2, despite having the same ratings on the important N books. This is because the 
# dimensionality of the space matters. 
#
# Then, I think it is better to consider the target_books, not only because computationally is
# highle cheaper, but also because it makes more sense to me.

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


#######################################################################################################

# Load the data

books = pd.read_csv("data/Books_cleaned.csv").drop('Unnamed: 0', axis = 1)

ratings_files = [f'data/Ratings_cleaned_part_{i}.csv' for i in range(1,6+1)]
ratings_dfs = [pd.read_csv(file) for file in ratings_files]
ratings = pd.concat(ratings_dfs, ignore_index=True).drop('Unnamed: 0', axis = 1)

books_genres = pd.read_csv("data/Books_genres_cleaned.csv").drop('Unnamed: 0', axis = 1)
books_genres['Genres'] = books_genres['Genres'].apply(ast.literal_eval)
books_genres_list = pd.read_csv("data/Books_genres_list_cleaned.csv").drop('Unnamed: 0', axis = 1)

#######################################################################################################

# Dash application

# Maximum number of users with coincidences that we use
n_users_upper_limit = 1000

# Number of neighbours
default_number_neighbours = 500

# Create a client of Google Cloud Storage
storage_client = storage.Client()
# Name of the bucket of Google Cloud Storage to store the user files with their selection
bucket_name = "user-files-bucket"


# Create a dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


###############################################################################
#                                                                             #
#                                   lAYOUT                                    #
#                                                                             #
###############################################################################


# Create an app layout
app.layout = html.Div([
    dcc.Store( # Store to maintain app state
        id='app_state', 
        data={'initial_explanation_ongoing': True,
              'book_selection_ongoing': False,
              'potential_recommendations_ongoing': False,
              'final_recommendations_ongoing': False}
    ),  
    dcc.Store( # Store to store the ratings 
        id='rating_store'
    ),
    dcc.Store(
        id='potential_recommendations_df'
    ), 
    #
    # Initial explanation
    #
    html.Div([
        html.H1("Book Recommendation System"),
        html.P("Welcome to the books recommender! As the name indicates, here you can have book recommendations after following a few (straightforward) steps."),
        html.P("When you are ready to start, press the button below."),
        html.Button("Start the program!", id="start_book_selection_button"),
    ], id='initial_explanation', style={'display': 'block'}),
    #
    # Book Selection
    #
    html.Div([
        html.H1("Book selection"),
        html.P("If you have previously saved a selection or want to save your current one, you can enter your ID to load it."),
        dcc.Input(id='user_id_input', type='text', placeholder='Enter your user ID.'),
        html.Button("Load Ratings", id="load_ratings_button"),
        html.P('That user ID is not in our system. If it is your first time here, you can use this ID to save your first selection. Otherwise, try again with a valid value.', 
               id='nonexistent_userID', style={'display': 'none'}),
        html.P("Choose as many books as you want from the list and rate them. Select at least one."),
        dcc.Dropdown(
            id='dropdown_book_titles',
            options=[
                {'label': book_title, 'value': book_title} for book_title in books['Title']
            ],
            multi=True, # Allow multiple selection
            placeholder="Select books...",
            className='dropdown-hide-selected',
            style={'display': 'block'} # Default style to display the dropdown
        ),
        html.Button("Save Ratings", id="save_ratings_button"),
        html.Button("Finish selection", id="finish_book_selection_button"),  # Button to finish selection
        html.P(
            "No book selected! Please select at least one book.",
            id='text_no_select', 
            style={'display': 'none'}
        ),
        html.Div(id='selected_books_container'), # Container to show the selected books       
    ], id='book_selection', style={'display': 'none'}),
    #
    # Recommender program
    # 
    html.Div([
        html.H1("Obtaining your recommendations"),
        html.P('Wait while the recommendations are obtained...')
    ], id='potential_recommendations_program', style={'display': 'none'}), 
    #
    # Final recommendations
    # 
    html.Div([
            html.Div([
                html.H1("Here are your recommendations!"),
            html.Div([
                html.P('Your recommendations are ready for you. You can indicate in the slider how many books do you want to see in the list: '),
                dcc.Slider(
                    id='number_recom_slider',
                    min=1,
                    max=20,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(1, 21)}
                ),
            ]),
            html.P('If you want your recommendations to satisfy any genre selection, please, select the genres in the dropdown below.'),
            html.Div([
                html.P('Do you want the recommendations to include all the selected genres or just any of them?', 
                       style={'margin-left': '30px'}),
                html.Button("All", id="include_all_genres", n_clicks=0, style={'margin-left': '30px', 'margin-right': '15px'}),
                html.Button("Any", id="include_any_genres", n_clicks=0),
                dcc.Store(
                    id='genre_button_state', 
                    data={'include_all_genres': False, 
                          'include_any_genres': True,
                          'have_they_changed': False}
                ),
            ], style={'display': 'flex', 'align-items': 'center'}),
            html.Div([
                dcc.Dropdown(
                    id='dropdown_include_genres',
                    multi=True,
                    placeholder="Select genre(s) to include..."
                )
            ])
        ], style={'display': 'block'}),
        html.Div([
            html.P("If you want your recommendations to exclude any genre selection, please, select the genres in the dropdown below."),
            html.Div([
                dcc.Dropdown(
                    id='dropdown_exclude_genres',
                    multi=True,
                    placeholder="Select genre(s) to exclude..."
                )
            ])
        ], style={'display': 'block'}),
        html.P("Note: Both dropdowns only include genres that are present in your recommendations."),
        html.P(
            "No recommendations available with your genre selection. Please, change your choice.", 
            id='text_no_recommendations', 
            style={'display': 'none'}
        ),
        html.H2('Your recommendations:'),
        html.Div(id='recommended_books_container')
    ], id='final_recommendations', style={'display': 'none'})
])


###############################################################################
#                                                                             #
#                            UPDATE THE APP STATE                             #
#                                                                             #
###############################################################################


# Callback to show/hide components based on app state
@app.callback(
    [Output('initial_explanation', 'style'),
     Output('book_selection', 'style'),
     Output('potential_recommendations_program', 'style'),
     Output('final_recommendations', 'style')],
    [Input('app_state', 'data')]
)
def update_components_visibility(app_state):
    initial_explanation_style = {'display': 'block'} if app_state['initial_explanation_ongoing'] else {'display': 'none'}
    book_selection_style = {'display': 'block'} if app_state['book_selection_ongoing'] else {'display': 'none'}
    recommendations_program_style = {'display': 'block'} if app_state['potential_recommendations_ongoing'] else {'display': 'none'}
    final_recommendations_style = {'display': 'block'} if app_state['final_recommendations_ongoing'] else {'display': 'none'}
    
    return initial_explanation_style, book_selection_style, recommendations_program_style, final_recommendations_style


###############################################################################
#                                                                             #
#                            INITIAL EXPLANATION                              #
#                                                                             #
###############################################################################


# Callback to update app state when the start program is clicked
@app.callback(
     Output('app_state', 'data', allow_duplicate=True),
    [Input('start_book_selection_button', 'n_clicks')],
     State('app_state', 'data'),
     prevent_initial_call=True
)
def update_app_state(n_clicks, app_state):
    if n_clicks is not None:
        app_state['initial_explanation_ongoing'] = False
        app_state['book_selection_ongoing'] = True
    return app_state
    

###############################################################################
#                                                                             #
#                               BOOK SELECTION                                #
#                                                                             #
###############################################################################


# Callback to load a previous selection of a user
@app.callback(
    [Output('rating_store', 'data', allow_duplicate=True),
     Output('dropdown_book_titles', 'value', allow_duplicate=True),
     Output('nonexistent_userID', 'style')],
    [Input('load_ratings_button', 'n_clicks')],
    [State('user_id_input', 'value')],
    prevent_initial_call=True
)
def load_ratings(n_clicks, user_id):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    user_file = f'user_files/user_ratings_{user_id}.json'
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            rating_store = json.load(f)
        selected_books = list(rating_store.keys())
        return rating_store, selected_books, {'display': 'none'}
    else:
        return {}, [], {'display': 'block', 'fontSize': 15, 'color': 'red'}


# Callback to update app state when finish button is clicked and to hide the "No book selected!" message
@app.callback(
    [Output('app_state', 'data'),
     Output('text_no_select', 'style')],
    [Input('finish_book_selection_button', 'n_clicks'),
     Input('dropdown_book_titles', 'value')],
     State('app_state', 'data')
)
def update_app_state_or_hide_message(n_clicks,  selected_books, app_state):
    ctx = dash.callback_context

    # Determine which input triggered the callback
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'finish_book_selection_button':
        # This branch handles the finish_book_selection_button changes
        if n_clicks is not None:
            if not selected_books:
                text_no_select_style = {'display': 'block'}
            else:
                text_no_select_style = {'display': 'none'}
                app_state['book_selection_ongoing'] = False
                app_state['potential_recommendations_ongoing'] = True
        return app_state, text_no_select_style
    else:
        if trigger_id == 'dropdown_book_titles':
            # This branch handles the dropdown selection changes
            return dash.no_update, {'display': 'none'}  


# Callback to display the selected books by the user from the initial dropdown
@app.callback(
    Output('selected_books_container', 'children'),
    [Input('dropdown_book_titles', 'value')],
    [State('rating_store', 'data')] # State is used to access the current state of a component without triggering the callback
)
def display_selected_books(selected_books, rating_store):
    if selected_books:
        books_info = []
        for book_title in selected_books:
            book_row = books[books['Title'] == book_title].iloc[0]
            image_url = book_row['Image_url']
            author = book_row['Authors']
            rating_value = rating_store.get(book_title, 1) if rating_store else 1 # 1 is the default (and minimum) rating
            rating = dash_grocery.Stars(
                id={'type': 'rating', 'index': book_title}, 
                count=5, value=rating_value, color2="gold", size=30, edit=True, half=False
            )
            book_info = html.Div([
                html.Div([
                    html.Button('x', id={'type': 'remove_book_dropdown', 'index': book_title}, n_clicks=0, style={'margin-right': '10px'}),
                    html.Img(src=image_url, style={'width': '70px', 'height': '100px', 'margin-top': '10px', 'margin-right': '20px'}),
                    html.Div([
                        html.H3(book_title, style={'margin-right': '20px'}),
                        html.H4(author, style={'margin-right': '20px'}),
                    ]),
                    rating
                ], style={'display': 'flex', 'align-items': 'center'}),
            ])
            books_info.append(book_info)
        return books_info
    else:
        return html.Div()


# Callback to handle book removal using the 'x' button
@app.callback(
    Output('dropdown_book_titles', 'value'),
    [Input({'type': 'remove_book_dropdown', 'index': ALL}, 'n_clicks')],
    [State('dropdown_book_titles', 'value')]
)
def remove_selected_book_from_dropdown(n_clicks, selected_books):
    # This allows to access detailed information about what has actuvated a 
    # callback and about the inputs and outputs involved in the function
    ctx = dash.callback_context 

    # ctx.triggered is a list of the inputs that activated the callback
    # Each element is a dictionary with the keys 'prop_id' and 'value'
    if not ctx.triggered: 
        raise dash.exceptions.PreventUpdate

    # Determine which input triggered the callback
    # 'prop_id' indicates what input changed
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_id = ast.literal_eval(trigger_id)

    for i, elem in enumerate(selected_books):
        if elem == trigger_id['index'] and n_clicks[i] != 0:
            book_to_remove = elem
            if book_to_remove in selected_books:
                selected_books.remove(book_to_remove)
                return selected_books

    raise dash.exceptions.PreventUpdate


# Callback to update the Store with the values of the ratings
@app.callback(
    Output('rating_store', 'data'),
    [Input({'type': 'rating', 'index': ALL}, 'value')],  # Dynamic input for all the ratings
    [State('dropdown_book_titles', 'value'),  # State for the selected books
     State('rating_store', 'data')]  # Access to the current Store  
)
def update_rating_store(rating_values, selected_books, rating_store):
    # To initialize the store (dictionary) every time the function is called.
    # This guarantees that the books that were removed are dropped from the dictionary
    rating_store = {}

    # If there are no books selected, exit the function
    if selected_books is None:
        return rating_store
    
    # Iterate over the selected books and their corresponding rating values
    for book_title, rating_value in zip(selected_books, rating_values):
        # Update the rating value for each selected book
        rating_store[book_title] = rating_value

    # Save the dictionary of selected books
#    with open('rating_store.json', 'w') as f:
#            json.dump(rating_store, f)
    
    return rating_store


# Callback to save the user selection
#@app.callback(
#    Output('rating_store', 'data', allow_duplicate=True),
#    [Input('save_ratings_button', 'n_clicks')],
#    [State('user_id_input', 'value'), 
#     State('rating_store', 'data')],
#    prevent_initial_call=True
#)
#def save_ratings(n_clicks, user_id, rating_store):
#    if n_clicks is None:
#        raise dash.exceptions.PreventUpdate

#    user_file = f'user_files/user_ratings_{user_id}.json'
#    with open(user_file, 'w') as f:
#        json.dump(rating_store, f)
#    return rating_store


###############################################################################
#                                                                             #
#                            RECOMMENDATION SYSTEM                            #
#                                                                             #
###############################################################################


@app.callback(
    [Output('potential_recommendations_df', 'data'),
     Output('app_state', 'data', allow_duplicate=True)],
    [Input('app_state', 'data')],
    [State('rating_store', 'data')],
     prevent_initial_call=True
)
def update_intermediate_state(app_state, rating_store):
    if not app_state['book_selection_ongoing'] and app_state['potential_recommendations_ongoing']:

        target_UserID = 19960808 # This value is arbitrary, but not an existing UserID

        # ratings dataframe including the target user ratings
        ratings_new = user_dictionary_to_df(rating_store, target_UserID, books, ratings)

        # Books rated by the target user
        target_books = ratings_new[ratings_new['UserID'] == target_UserID].BookID.values

        # Selected users to get the recommendations
        selected_users, selected_ratings = selected_users_df(ratings_new, target_books, n_users_upper_limit, target_UserID)

        # Creating the matrix with users and books
        #ratings_csr_matrix, users = get_users_matrix(selected_ratings)
        ratings_csr_matrix, users = get_users_matrix(selected_ratings, target_books)

        # Get the potential recommendations
        potential_recommendations = knn_model(ratings_csr_matrix, users, target_UserID, default_number_neighbours, selected_ratings, target_books)   
        potential_recommendations_json = potential_recommendations.to_json(orient='split')
    
        # Save the table of potential recommendations
#        potential_recommendations_list = potential_recommendations.to_dict(orient='records')
#        with open('potential_recommendations.json', 'w') as f:
#            json.dump(potential_recommendations_list, f)

        # Update the state to indicate that the process has finished
        app_state['potential_recommendations_ongoing'] = False
        app_state['final_recommendations_ongoing'] = True
        
        return potential_recommendations_json, app_state

    else:
        raise dash.exceptions.PreventUpdate


###############################################################################
#                                                                             #
#                            FINAL RECOMMENDATIONS                            #
#                                                                             #
###############################################################################


# Callback to modify the genres options of the dropdown that the recommended books must satisfy
@app.callback(
    [Output('dropdown_include_genres', 'options'),
     Output('dropdown_include_genres', 'value'),
     Output('dropdown_exclude_genres', 'options'),
     Output('dropdown_exclude_genres', 'value'),
     Output('genre_button_state', 'data', allow_duplicate=True)],
    [Input('app_state', 'data'),
     Input('potential_recommendations_df' , 'data'),
     Input('dropdown_include_genres', 'value'),
     Input('dropdown_exclude_genres', 'value'),
     Input('genre_button_state', 'data')],
     prevent_initial_call=True
)
def get_genres_to_include(app_state, pot_recom_json, selected_included_genres, selected_excluded_genres, button_state):
    # pot_recom_json : all the potential recommendations for the user
    # selected_included_genres : genres currently selected in the included genres dropdown
    # selected_excluded_genres : genres currently selected in the excluded genres dropdown
    # button_state : dictionary with the state of the All and Any buttuns
    
    if pot_recom_json is None or not app_state['final_recommendations_ongoing']:
        raise dash.exceptions.PreventUpdate
    
    pot_recom = pd.read_json(StringIO(pot_recom_json), orient='split')

    # Include the genres lists in the dataframe
    pot_recom = pd.merge(pot_recom, books_genres[['BookID', 'Genres', 'Genre_1', 'Genre_2', 'Genre_3', 'Genre_4', 'Genre_5', 'Genre_6', 'Genre_7']], on='BookID', how='left')

    # Already selected excluded genres
    if selected_excluded_genres is None:
        excluded_genres = []
    else:
        excluded_genres = [genre for genre in selected_excluded_genres]

    # Already selected included genres
    if selected_included_genres is None:
        included_genres = []
    else:
        included_genres = [genre for genre in selected_included_genres]

    # Keep only the books that do not have the excluded genres
    exclude_mask = np.logical_not(np.logical_or.reduce([pot_recom[f'Genre_{i}'].isin(excluded_genres) for i in range(1, 8)]))
    pot_recom = pot_recom
    
    # If the state of the buttons has just changed, initialize the selected included genres
    if button_state['have_they_changed'] == True:
        included_genres = []
        # Put the have_they_changed state in the genre button state back to False
        button_state['have_they_changed'] = False

    # List with all the lists of genres of the potential recommendations. The array is also converted to a list
    lists_genres = pot_recom[['Genres']].values
    lists_genres = [item[0] for item in lists_genres]
    
    # The list for the dropdown depends on the genre buttons selection
    if button_state['include_all_genres'] == True:
        # Lists that include the selected genres
        filtered_lists_genres = [lst for lst in lists_genres if all(genre in lst for genre in included_genres)] 
    else:
        # Lists that include the selected genres
        if not included_genres:
            filtered_lists_genres = lists_genres
        else:
            filtered_lists_genres = [lst for lst in lists_genres if any(genre in lst for genre in included_genres)] 

    # One list with all the genres of the previous lists
    possible_genres = list(itertools.chain(*filtered_lists_genres)) 
    # Drop duplicates
    include_list_for_dropdowns = list(set(possible_genres))

    # The list for the excluded genres has to include the excluded genres too for them to remain selected
    for genre in included_genres:
        if not genre in include_list_for_dropdowns:
            include_list_for_dropdowns.append(genre)
    
    # The list for the excluded genres has to include the excluded genres too for them to remain selected
    # Also, it has to exclude the selected genres to be included
    exclude_list_for_dropdowns = include_list_for_dropdowns.copy()
    for genre in excluded_genres:
        exclude_list_for_dropdowns.append(genre)
    for genre in included_genres:
        if genre in exclude_list_for_dropdowns:
            exclude_list_for_dropdowns.remove(genre)

    # Remove 'Empty' from the genres lists
    empty = 'Empty'
    if empty in include_list_for_dropdowns:
        include_list_for_dropdowns.remove(empty)
    if empty in exclude_list_for_dropdowns:
        exclude_list_for_dropdowns.remove(empty)

    # Sort the elements in the lists alphabetically
    include_list_for_dropdowns.sort()
    exclude_list_for_dropdowns.sort()

    # Options for the dropdowns
    options_include = [
        {'label': genre, 'value': genre} for genre in include_list_for_dropdowns
    ]

    options_exclude = [
        {'label': genre, 'value': genre} for genre in exclude_list_for_dropdowns
    ]
    
    return options_include, included_genres, options_exclude, excluded_genres, button_state


# Callback to update the state of the all/any genres button
@app.callback(
    [Output('genre_button_state', 'data'),
     Output('include_all_genres', 'style'),
     Output('include_any_genres', 'style')],
    [Input('include_all_genres', 'n_clicks'), 
     Input('include_any_genres', 'n_clicks')],
    [State('genre_button_state', 'data')]
)
def toggle_genre_button_and_style(button_all_clicks, button_any_clicks, button_state):
    changed_id = [trigger_id['prop_id'] for trigger_id in dash.callback_context.triggered][0]
    
    # Update the state of the buttons
    if 'include_all_genres' in changed_id:
        if button_state['include_all_genres'] == True:
            button_state['have_they_changed'] = False
        else:
            button_state['have_they_changed'] = True
        button_state['include_all_genres'] = True
        button_state['include_any_genres'] = False
    elif 'include_any_genres' in changed_id:
        if button_state['include_any_genres'] == True:
            button_state['have_they_changed'] = False
        else:
            button_state['have_they_changed'] = True
        button_state['include_all_genres'] = False
        button_state['include_any_genres'] = True
    
    # Update the style of the buttons depending on the state
    button_all_style = {'background-color': 'blue', 'color': 'white', 'margin-left': '30px', 'margin-right': '15px'} if button_state['include_all_genres'] else {'margin-left': '30px', 'margin-right': '15px'}
    button_any_style = {'background-color': 'blue', 'color': 'white'} if button_state['include_any_genres'] else {}
    
    return button_state, button_all_style, button_any_style


# Callback to print the recommendations
@app.callback(
    [Output('recommended_books_container', 'children'),
     Output('text_no_recommendations', 'style')],
    [Input('app_state', 'data'),
     Input('potential_recommendations_df' , 'data'),
     Input('dropdown_include_genres', 'value'),
     Input('dropdown_exclude_genres', 'value'),
     Input('number_recom_slider', 'value')],
     State('genre_button_state', 'data')
)
def get_the_final_recommendations(app_state, pot_recom_json, selected_genres, excluded_genres, num_recom, button_state):
    if pot_recom_json is None or not app_state['final_recommendations_ongoing']:
        raise dash.exceptions.PreventUpdate
        
    # Genres selected for the books to include or exclude them
    included_genres = selected_genres if selected_genres else []
    excluded_genres = excluded_genres if excluded_genres else []

    # Potential book recommendation
    pot_recom = pd.read_json(StringIO(pot_recom_json), orient='split')
   
    # Filter the potential recommendations by the selected genres
    if button_state['include_all_genres'] == True:
        combine = True
    else:
        combine = False
    recommendations = books_satisfying_genres(pot_recom, books_genres, included_genres, excluded_genres, combine=combine)
    recommendations = pd.merge(recommendations, books[['BookID', 'Authors', 'ISBN', 'Title', 'Average_Rating', 'Image_url']], on='BookID', how='left')
    
    # Number of recommendations
    recommendations = recommendations.head(num_recom)
    
    # Save the table of potential recommendations
    recommendations_list = recommendations.to_dict(orient='records')
#    with open('recommendations.json', 'w') as f:
#        json.dump(recommendations_list, f)

    # Crear la lista de recomendaciones para mostrar en el contenedor
    recommendations_display = []
    idx = 1
    for rec in recommendations_list:
        book_title = rec['Title']
        author = rec['Authors']
        isbn = rec['ISBN']
        rating = rec['Average_Rating']
        book_image_url = rec['Image_url']
        recommendations_display.append(
            html.Div([
                html.H3(ordinal_number(idx) + ' recomendation:'),
                html.Div([
                    html.Img(src=book_image_url, style={'width': '93px', 'height': '130px', 'margin-right': '20px'}),
                    html.Div([
                        html.P(book_title, style={'fontSize': 18, "font-weight": "bold"}),
                        html.P('Author: ' + author, style={'fontSize': 15}),
                        html.P('Goodreads rating: ' + str(rating), style={'fontSize': 15}),
                        html.P('ISBN: ' + isbn, style={'fontSize': 15}),
                    ], style={'flex': '1', 'margin-bottom': '10px'}),
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px', 'margin-left': '40px'})
            ])
        )
        idx += 1

    # Show or hide the 'No recommendations' message
    if len(recommendations_list) == 0:
        text_no_recommendations_style = {'display': 'block', 'fontSize': 20, 'color': 'red'}
    else:
        text_no_recommendations_style = {'display': 'none', 'fontSize': 20, 'color': 'red'}

    return recommendations_display, text_no_recommendations_style
    
    
###############################################################################
#                                                                             #
#                          FOR GOOGLE CLOUD PLATFORM                          #
#                                                                             #
###############################################################################    
    
   
# Callback to store the users selections
@app.callback(
    Output('rating_store', 'data', allow_duplicate=True),
    [Input('save_ratings_button', 'n_clicks')],
    [State('user_id_input', 'value'), 
     State('rating_store', 'data')],
     prevent_initial_call=True
)
def save_ratings_in_cloud(n_clicks, user_id, rating_store):
    if n_clicks is None:
       raise dash.exceptions.PreventUpdate

    # Convert the user selection into a JSON file
    rating_store_json = json.dumps(rating_store)

    # Name of the file in Google Cloud Storage
    blob_name = f"user_ratings_{user_id}.json"

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob (file) in the bucket
    blob = bucket.blob(blob_name)

    # Upload the JSON file to Google Cloud Storage
    blob.upload_from_string(rating_store_json)

    return rating_store
    

if __name__ == '__main__':
    app.server(host='0.0.0.0', port=8080, debug=True)
    
#######################################################################################################
