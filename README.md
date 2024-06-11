
#  BOOK RECOMMENDATION SYSTEM 

### Author: Pablo Escribano
### Date of last update: 06/06/2024 


This is a python program to get book recommendations based on user feedback.



Note: Instead of using the files tags and book_tags, I did web scrapping to get the genres of the books. Although there is more information in the original files, I think it is enough with having 7 genres of each book and we avoid having many many tags which are duplicated or even that are irrelevant.





## Dash application using Google Cloud Platform

1. Create a Google Cloud Platform account: You can do it through this url: https://cloud.google.com/
2. Create a new project: Once you are logged in, go to 'Console' and create a new project.
3. Install the Google Cloud SDK: Follow the steps from this url: https://cloud.google.com/sdk/docs/install-sdk?hl=es-419#deb
4. Configure the project locally: Ensure the Dash app is ready to be displayed.
5. Configure the application for the App Engine: Create the file 'app.yaml' in the root directory of the project. This file specifies how the application will execute in App Engine. A basic example for the file is shown here:

runtime: python39
instance_class: F1
entrypoint: gunicorn -b :$PORT app:server

6. Deploy your application in App Engine: Use the Google Cloud SDK to deploy your application in App Engine. From the directory of the project, run in the terminal:

gcloud app deploy

7. Verify the deployment: Once the deployment has finished, you should be able to access your Dash application through the url given by the GCP.


