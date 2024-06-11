#############################################################################
#                                                                           #
#            DASH APPLICATION FOR THE BOOK RECOMMENDATION SYSTEM            #
#                                                                           #
#############################################################################



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

# Import my functions
from my_functions import *



# Load the data

books = pd.read_csv("data/Books_cleaned.csv").drop('Unnamed: 0', axis = 1)

ratings_files = [f'data/Ratings_cleaned_part_{i}.csv' for i in range(1,6+1)]
ratings_dfs = [pd.read_csv(file) for file in ratings_files]
ratings = pd.concat(ratings_dfs, ignore_index=True).drop('Unnamed: 0', axis = 1)

books_genres = pd.read_csv("data/Books_genres_cleaned.csv").drop('Unnamed: 0', axis = 1)
books_genres_list = pd.read_csv("data/Books_genres_list_cleaned.csv").drop('Unnamed: 0', axis = 1)



# Dash application

# Maximum number of users with coincidences that we use
n_users_upper_limit = 10000 

# Number of neighbours
default_number_neighbours = 50


# Create a dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)


###############################################################################
#                                                                             #
#                                   lAYOUT                                    #
#                                                                             #
###############################################################################


# Create an app layout
app.layout = html.Div([
    dcc.Store( # Store to maintain app state
        id='app_state', 
        data={'book_selection_ongoing': True,
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
    ], id='book_selection', style={'display': 'block'}),
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
        html.H1("Here are your recommendations!"),
        html.Div([
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
    [Output('book_selection', 'style'),
     Output('potential_recommendations_program', 'style'),
     Output('final_recommendations', 'style')],
    [Input('app_state', 'data')]
)
def update_components_visibility(app_state):
    book_selection_style = {'display': 'block'} if app_state['book_selection_ongoing'] else {'display': 'none'}
    recommendations_program_style = {'display': 'block'} if app_state['potential_recommendations_ongoing'] else {'display': 'none'}
    final_recommendations_style = {'display': 'block'} if app_state['final_recommendations_ongoing'] else {'display': 'none'}
    
    return book_selection_style, recommendations_program_style, final_recommendations_style


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
            rating_value = rating_store.get(book_title, 1) if rating_store else 1 # 1 is the default (and minimum) rating
            rating = dash_grocery.Stars(
                id={'type': 'rating', 'index': book_title}, 
                count=5, value=rating_value, color2="gold", size=30, edit=True, half=False
            )
            book_info = html.Div([
                html.Div([
                    html.Button('x', id={'type': 'remove_book_dropdown', 'index': book_title}, n_clicks=0, style={'margin-right': '10px'}),
                    html.Img(src=image_url, style={'width': '50px', 'height': '75px', 'margin-top': '10px', 'margin-right': '20px'}),
                    html.H3(book_title, style={'margin-right': '20px'}),
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
@app.callback(
    Output('rating_store', 'data', allow_duplicate=True),
    [Input('save_ratings_button', 'n_clicks')],
    [State('user_id_input', 'value'), 
     State('rating_store', 'data')],
    prevent_initial_call=True
)
def save_ratings(n_clicks, user_id, rating_store):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    user_file = f'user_files/user_ratings_{user_id}.json'
    with open(user_file, 'w') as f:
        json.dump(rating_store, f)
    return rating_store


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
        ratings_csr_matrix, ratings_matrix = get_users_matrix(selected_ratings)

        # Get the potential recommendations
        potential_recommendations = knn_model(ratings_csr_matrix, ratings_matrix, target_UserID, default_number_neighbours, selected_ratings, target_books)
                                           
        del ratings_matrix
    
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
    pot_recom = pot_recom[~pot_recom.apply(lambda row: contains_any_genre(row, excluded_genres), axis=1)]

    # If the state of the buttons has just changed, initialize the selected included genres
    if button_state['have_they_changed'] == True:
        included_genres = []
        # Put the have_they_changed state in the genre button state back to False
        button_state['have_they_changed'] = False

    # List with all the lists of genres of the potential recommendations. The array is also converted to a list
    lists_genres = pot_recom[['Genres']].values
    lists_genres = [ast.literal_eval(item[0]) for item in lists_genres]
    
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

    # Options for the dropdowns
    options_include = [
        {'label': genre, 'value': genre} for genre in include_list_for_dropdowns
    ]

    options_exclude = [
        {'label': genre, 'value': genre} for genre in exclude_list_for_dropdowns
    ]
    
    return options_include, included_genres, options_exclude, excluded_genres, button_state


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
     Input('dropdown_exclude_genres', 'value')],
     State('genre_button_state', 'data')
)
def get_the_final_recommendations(app_state, pot_recom_json, selected_genres, excluded_genres, button_state):
    if pot_recom_json is None or not app_state['final_recommendations_ongoing']:
        raise dash.exceptions.PreventUpdate
        
    # Genres selected for the books to include or exclude them
    included_genres = selected_genres if selected_genres else []
    excluded_genres = excluded_genres if excluded_genres else []

    # Potential book recommendation
    pot_recom = pd.read_json(StringIO(pot_recom_json), orient='split')
    # Include the genres lists in the dataframe
    pot_recom = pd.merge(pot_recom, books_genres[['BookID', 'Genres']], on='BookID', how='left')

    # Filter the potential recommendations by the selected genres
    if button_state['include_all_genres'] == True:
        combine = True
    else:
        combine = False
    recommendations = books_satisfying_genres(pot_recom, books_genres, included_genres, excluded_genres, combine=combine)
    recommendations = pd.merge(recommendations, books[['BookID', 'Title', 'Image_url']], on='BookID', how='left')

    # Number of recommendations
    n = 10
    recommendations = recommendations.head(n)
    
    # Save the table of potential recommendations
    recommendations_list = recommendations.to_dict(orient='records')
#    with open('recommendations.json', 'w') as f:
#        json.dump(recommendations_list, f)

    # Crear la lista de recomendaciones para mostrar en el contenedor
    recommendations_display = []
    for rec in recommendations_list:
        book_title = rec['Title']
        book_image_url = rec['Image_url']
        recommendations_display.append(
            html.Div([
                html.Img(src=book_image_url, style={'width': '50px', 'height': '75px', 'margin-right': '20px'}),
                html.H4(book_title, style={'margin-right': '20px'})
            ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'})
        )

    # Show or hide the 'No recommendations' message
    if len(recommendations_list) == 0:
        text_no_recommendations_style = {'display': 'block', 'fontSize': 20, 'color': 'red'}
    else:
        text_no_recommendations_style = {'display': 'none', 'fontSize': 20, 'color': 'red'}

    return recommendations_display, text_no_recommendations_style


if __name__ == '__main__':
    app.run_server(debug=True)
