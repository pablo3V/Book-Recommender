
#  BOOK RECOMMENDATION SYSTEM 

### Author: Pablo Escribano
### Date of last update: 18/06/2024 


After finishing a captivating book, it is common to experience a sense of loss, as if saying goodbye to a friend. The search for a good book to fill that void can be intimidating, with the worry that nothing else will live up to its predecessor. This is where an application designed to recommend books can offer a beacon of hope in the form of carefully selected recommendations.

Imagine a program that understands your reading preferences, knows your favorite genres and authors, and can recommend titles with captivating storytelling. Such an app could become a trusted companion for book lovers, making the transition from one book to the next simpler and more satisfying. This is precisely what this recommender aims to achieve.

While there is still plenty of work to be done, this Python program is already capable of providing book recommendations based on user feedback.

The project is split in several parts:

- [Part I - Data Wrangling](1-Data_Wrangling.ipynb)
- [Part II - Exploratory Data Analysis (EDA)](2-EDA.ipynb)
- [Part III - Collaborative Filtering](3-Model.ipynb)
- [Part IV - Dash Application](4-Dash.ipynb)


## Part I - Data Wrangling

### The Data

#### The goodbooks-10k repository

First of all, the data that we will use in this project is collected. The datasets are downloaded from the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) GitHub repository. It contains approximately six million user ratings for ten thousand popular books from Goodreads. The original datasets are placed in the [data_preprocessed/](data_preprocessed/) directory. There are four files there:

- [books](data_preprocessed/books.csv): Includes the metadata for each of the 10K books.
- [ratings](data_preprocessed/ratings_part_1.csv): Each row represent the rating for a book with a given bookID by a user with a given userID. The dataset has been divided into four smaller files, with the help of the program [split_csv.py](split_csv.py), to avoid GitHub warnings.
- [book_tags](data_preprocessed/book_tags.csv): Contains tags, shelves, and genres assigned by Goodreads users to the books.
- [tags](data_preprocessed/tags.csv): Translates the tags IDs to names.

However, in this project, we decided not to use the book_tags and tags files, as they contain a lot of information that is not relevant for our purposes. Additionally, many tags for the same genres have different names, making the process of cleaning the dataset difficult and tedious.

#### Web scraping

As mentioned, we did not obtain the genres of the books from the book_tags and tags files, despite their importance. Instead, we performed web scrapping. Additionally, many books were missing cover image URLs, ISBNs, or publication years. Using the notebooks in the [web_scraping](web_scraping/) folder, we succesfully obtained all these features.

A final remark regarding the genres of the books: While the original files contain more information, having seven genres per book should be sufficient for our purposes. These seven genres are easy to scrape and we avoid many tags that are either duplicated or irrelevant.

### Data Cleaning

Once all the data is collected, it needs to be processed. In the rest of this section, we handle common issues associated with raw (preprocessed) data, including missing values, duplicates, and outliers. 

Additional considerations:

- We dropped all books with titles containing Arabic characters for simplicity.
- We excluded users with fewer than 10 ratings to ensure statistical significance.
- Some books in the dataset are collections of other books included in the set. To avoid recommending a collection when the individual books have already been read, we assign the individual books the rating of the collection and then drop the collection from the datasets.
- We added additional columns to the books file, including the name of the saga and the volume, if applicable.

Note: Once the data is processed and saved, we use the [split_csv.py](split_csv.py) program to divide the Ratings_cleaned.csv file into smaller parts.


## Part II - Exploratory Data Analysis (EDA)

In this part of the project, we identified interesting aspects of the books datasets. A a detailed analysis can be found in the [2-EDA.ipynb](2-EDA.ipynb) notebook. The main conclusion is that the impact of numerical features on book ratings is rather small, suggesting that book ratings are primarily driven by other factors, such as the quality of the books.


## Part III - Collaborative Filtering

Collaborative filtering is a well-known method for making recommendations, with user-based collaborative filtering being a prime example. Imagine you are looking for a new book to read but you are unsure which one to choose. If you have friends or relatives whose taste in books aligns well with yours, asking them for recommendations makes sense. This idea is the foundation of user-based collaborative filtering.

Here is how it works: 

1. First, you identify other users who have similar tastes to the target user based on their ratings of the same set of books. For example, if you enjoyed all of Brandon Sanderson's books, you look for users who also liked those books.

2. Once you have found these similar users, you take their average ratings for books that the target user has not read yet. For instance, you check how these Brandon Sanderson fans rated other books.

3. Finally, you recomment the books with the highest average ratings to the current user. 

These three steps form the core of the user-based collaborative filtering algorithm.

Before implementing this algorithm, we need to restructure our data. For this method, data is typically structured such that each row corresponds to a user and each column corresponds to a product (a book in our scenario).

### Steps Followed in the Notebook

1. **Create a DataFrame with Target User Rating**: We creare a DataFrame with the ratings of the target user for whom we are obtaining recommendations. These ratings are merged with the original ratings.

2. **Select Relevant Users**: To make the computation less demanding and faster, we select only the users that have rated at least ten of the books that the target user has rated. This way, from the original 53346 users in the dataset, we use only 819 of them.

3. **Construct User-Book Matrix**: We construct a matrix with these users (and the target user) where rows and columns correspond to users and books, respectively. However, the matrix we get has 4902534 elements, so it is better to use a sparse matrix, which only keeps the non-zero elements, leaving just 106903. This reflects the importance of using a sparse matrix. Notice how much larger the total size of the array is compared to the number of non-zero elements, which is just the number of ratings of the selected users. Sparse arrays/matrices allow us to represent these objects without explicitly storing all the 0-valued elements. This means that if the transactional data can be loaded into memory, the sparse array will fit in memory as well.

4. **Normalize the Sparse Matrix**: The sparse matrix is normalized since normalization can help mitigate the biases of users who may have different rating scales. Some common normalization techniques are:

    * **Mean-Centering** (User Mean Normalization): For each user's rating, subtract the user's average rating from each of their ratings. This helps to account for differences in user rating scales.

        r_{ui}' = r_{ui} - \bar{r}_u

        where r_{ui} is the original rating by user \( u \) for item \( i \), and \bar{r}_u is the average rating given by user \( u \).

    * **Z-score Normalization**: This method involves scaling the ratings based on the user's mean and standard deviation. It helps address both the mean and variance differences in user ratings.

        r_{ui}' = (r_{ui} - \bar{r}_u) ÷ sigma_u

        where sigma_u  is the standard deviation of the user's ratings.

    * **Min-Max Normalization**: This technique scales the ratings to a fixed range, typically [0, 1] or [-1, 1].

        r_{ui}' = (r_{ui} - \min(r_u)) ÷ (\max(r_u) - \min(r_u))

        where \min(r_u) and \max(r_u) are the minimum and maximum ratings given by user \( u \), respectively.

5. **Find Similar Users**: For all the users with at least 10 common ratings, we calculate the similarity of their ratings with the target user's ratings. Among the methods to calculate similarities, the cosine similarity and Pearson's correlation coefficient are the most popular. This step can be done either manually by coding everything step-by-step (as done in 'Step 2.1. Find similar users' of the notebook), or we can use a predefined function to get the similar users. This was done in 'Step 2.2. Find similar users - KNN algorithm', where we used the function 'NearestNeighbors' from sklearn.

6. **Generate Recommendations**: For the recommendations, we consider the books that the similar users have read but the target user has not yet read. We then compute the average rating for each of these books based on the ratings from these similar users. To make the average ratings more reliable, we will only include books that have been rated by at least 10 users. This process helps identify books that align with the target user's taste. 


## Part IV - Dash Application

In this final part of the project, we built a web application using Dash. The application follows the same steps outlined in the previous part. Note that several functions are defined in the Python files in the [my_functions](my_functions/) folder.

The layout of the Dash application is divided into four sections:

1. **The Initial Explanation**: This section provides a brief overview of the program.

2. **The Book Selection**: This is the first step to getting personalized book recommendations. Here, the users indicate which books they have previously read and and their ratings on a scale from 1 to 5. To do this, they simply write the name of the books in the dropdown menu and select the appropriate number of stars for their ratings. If they want to remove a selection, they have to click on the 'x' on the left side of the book cover. Additionally, the users can save their book selection or load a previously saved one by indicating a user ID.

3. **The Recommendations Process**: This is an intermediate section where the users wait while their personalized recommendations are being generated.

4. **The Final Recommendations**: In this last part of the program, the recommendations are displayed at the bottom of the page. The users can select how many recommendations to display, ranging from 1 to 20. Additionally, the users can specify genre restrictions for the recommendations. It is possible to include only books of certain genres or exclude books based on a genre selection. 

    Note that the genres listed in the dropdowns depend on the genres of the potential book recommendations for each user. Each time the potential recommendations change due to a genre specification, the dropdowns are updated to include only the relevant genres.
    
**Disclaimer**: There is still work to be done regarding the application's appearance. 

### Dash Application Using Google Cloud Platform

#### Important Considerations 

Once the Dash application is ready, we can deploy it online using Google Cloud Platform (GCP). Since we are using the free version, resources are limited, and certain considerations need to be taken into account:

- **Fewer Users with Coincidences**: Compared to the original Dash app, it is better to consider fewer users with coincidences with the target user. We set `n_users_upper_limit = 2000`.

- **Optimizing Potential Recommendations**: To speed up the process of generating potential recommendations, we construct the users-books matrix using only the books that the target user has rated. This significantly reduces the time required to create the matrix. This approach focuses on the importance of users sharing ratings on the same books as the target user. For example:

    * If the target user has rated N books, and user_1 has rated the same N books with the same ratings plus 1 additional book, while user_2 has rated the same N books with the same ratings plus 1000 additional books, user_1 will be closer to the target user according to the KNN algorithm. This is because the dimensionality of the space matters. 
    
    Although initially considered for the GCP version, this approach was implemented in the original Dash application as well.
    
- **Skipping `update_recommendations()` Function**: The function `update_recommendations()` is not used in the GCP app version because it causes the app to freeze and never return the recommendations. This function is intended to clean the potential recommendations by ensuring the user is recommended the next volume in a saga they have started (or the initial volume if they have not read anything), rather than random volumes out of order. Without this function, the user might get recommended volumes out of sequence.
    
- **Save/Load Book Selections Disabled**: Although implemented, the GCP version cannot save or load book selections due to resource constraints. These features work but take too long to compile, so the code is commented out.

#### Implementation 

With these considerations in mind, we followed the steps outlined in this [guide](https://datasciencecampus.github.io/deploy-dash-with-gcp/) to implement the GCP application. Here is a summary:

1. **Create your Dash Application**: Ensure the application runs locally and that you have the following files in the directory:

    * **main.py**: The Dash application and all the required functions in the same file.
    * **app.yaml*: Used to run the Dash app on GCP using gunicorn
    * **requirements.txt**: Includes all the packages needed to run the Dash app (make sure to include gunicorn).

2. **Create a Project on Google Cloud Platform**: Create a new project on GCP.

3. **Assign Project Ownership**: From 'Project info' -> 'Add people to this project', make yourself the owner of the project.

4. **Install gcloud**: Follow the indications here: [Install Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk?hl=es-419)

5. **Deploy Your Application Using gcloud Command Line Tool**: In the directory where you have cloned your GitHub repository, run:

```sh
gcloud config get-value project
```

to check the active project in gcloud. You can change the project with:

```sh
gcloud config set project project-id
```

Finally, deploy the app:

```sh
gcloud app deploy
```

Note: If you encounter issues as described here: [Gcloud App Deploy Error](https://stackoverflow.com/questions/64274811/gcloud-app-deploy-error-response-13-failed-to-create-cloud-build-invalid-buc), resolve them and re-run the command.

6. **Access the application**: Access the URL of your application via:

```sh
gcloud app browse
```

In our case, it is:

[https://book-recommendations-dash.ew.r.appspot.com](https://book-recommendations-dash.ew.r.appspot.com)








