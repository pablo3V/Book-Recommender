
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

However, before implementing this algorithm, we need to restructure our data. For this method, data is typically structured such that each rox corresponds to a user and each columns corresponds to a product (a book in our scenario).

We list here the main steps followed in the notebook with some comments:

1. We creare a DataFrame with the ratings of the target user for which we obtained the recommendations. This ratings are merged with the original ratings,

2. To make the computation less demanding and faster, we select only the users that have rated, at least, ten of the books that the target user has rated. This way, from the original 53346 users in the dataset, we only user 819 of them.

3. As mentioned before, we construct a matrix with this users (and the target user) whose rows and columns correspond to users and books, respectively. However, the matrix we get has 4902534 elements, so it is better to use a sparse matrix, which only keeps the non-zero elements, leaving just 106903. This reflects the importance of using a sparse matrix here. Notice how much larger the total size of the array is compared to the number of non-zero elements, which is just the number of ratings of the selected users. Sparse arrays/matrices allow us to represent these objects without explicitly storing all the 0-valued elements. This means that if the transactional data can be loaded into memory, the sparse array will fit in memory as well.

4. The sparse matrix is normalized, since normalization can help mitigating the biases of users who may have different rating scales. Some common normalization techniques are:

    * **Mean-Centering** (User Mean Normalization): For each user's rating, subtract the user's average rating from each of their ratings. This helps to account for differences in user rating scales.

    r_{ui}' = r_{ui} - \bar{r}_u

    where r_{ui} is the original rating by user \( u \) for item \( i \), and \bar{r}_u is the average rating given by user \( u \).

    * **Z-score Normalization**: This methid involves scaling the ratings based on the user's mean and standard deviation. It helps addressing both the mean and variance differences in user ratings.

    r_{ui}' = (r_{ui} - \bar{r}_u) ÷ sigma_u

    where sigma_u  is the standard deviation of the user's ratings.

    * **Min-Max Normalization**: This technique scales the ratings to a fixed range, typically [0, 1] or [-1, 1].

    r_{ui}' = (r_{ui} - \min(r_u)) ÷ (\max(r_u) - \min(r_u))

    where \min(r_u) and \max(r_u) are the minimum and maximum ratings given by user \( u \), respectively.

5. Find similar users. For all the users with at least 10 coincidences, we calculate the similarity of their ratings with the target user's ratings. Among the possibilities to calculate the similarities, the cosine similarity and Pearson's correlation coefficient are the most popular.











The funcion update_recommendations() is not used in the gcloud app version. This is due to the fact that, if done, the app freezes and never gives the recommendations.




## Dash application using Google Cloud Platform (GCP)

I followed the steps indicated here: https://datasciencecampus.github.io/deploy-dash-with-gcp/

To sum up:

1. Create your Dash Application: Once the application runs locally, make sure that you have the following files in the directory:

- main.py: the Dash application
- app.yaml: used to run the Dash app on GCP using gunicorn
- requirements.txt: includes all the packages needed to run the Dash app (make sure to include gunicorn)

2. Make a Project on Google Cloud Platform: Create a new project.

3. Make yourself the owner of the project: This can be done from 'Project info' -> 'Add people to this project'

4. Install gcloud: Follow the indications here: https://cloud.google.com/sdk/docs/install-sdk?hl=es-419

5. Deploy your Application using gcloud command line tool: In the directory where you have cloned your github repository, run the command:

gcloud config get-value project

to check which project is active in gcloud. You can change the project with the command:

gcloud config set project project-id

Finally, deploy the app:

gcloud app deploy

Note: I had the problem stated here: https://stackoverflow.com/questions/64274811/gcloud-app-deploy-error-response-13-failed-to-create-cloud-build-invalid-buc

Once it is solved, run the command again.

6. You can acces the url of your application via:

gcloud app browse

In my case, it is:

https://book-recommendations-dash.ew.r.appspot.com
