
#  BOOK RECOMMENDATION SYSTEM 

### Author: Pablo Escribano
### Date of last update: 06/06/2024 


This is a python program to get book recommendations based on user feedback.



Note: Instead of using the files tags and book_tags, I did web scrapping to get the genres of the books. Although there is more information in the original files, I think it is enough with having 7 genres of each book and we avoid having many many tags which are duplicated or even that are irrelevant.





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
