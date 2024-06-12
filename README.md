
#  BOOK RECOMMENDATION SYSTEM 

### Author: Pablo Escribano
### Date of last update: 06/06/2024 


This is a python program to get book recommendations based on user feedback.



Note: Instead of using the files tags and book_tags, I did web scrapping to get the genres of the books. Although there is more information in the original files, I think it is enough with having 7 genres of each book and we avoid having many many tags which are duplicated or even that are irrelevant.





## Dash application using Heraku

1. Prepare the Dash application: Make sure the Dash application is ready to be deployed. Test locally to ensure it works correctly.

2. Create an account on Heroku.

3. Install the Heroku CLI: Follow the instruction from https://devcenter.heroku.com/articles/heroku-cli to download and install the Heroku Command Line Interface (CLI). This allows you to deploy your application from the command line.

4. Log in to Heroku from the command line: Open a terminal and run the command 

heroku login

5. Create a new Heroku application: Create a new application on Heroku using the following command:

heroku create your-app-name

6. Configure the GitHub deployment: Configure automatic deployment from your GitHub repository. You can do this from the "Deploy" tab in your Heroku application's dashboard. Connect your GitHub account and select the repository containing your Dash application.

7. Select the deployment branch: In the deployment configuration on Heroku, choose the branch of your repository that you want to deploy (usually main or master).

8. Set up the requirements.txt file: Ensure you have a requirements.txt file in the root of your project listing all dependencies of your Dash application. This tells Heroku what Python packages need to be installed.

9. Set up the Procfile file: Create a file named Procfile in the root of your project (if you don't have one already) and make sure it contains a line specifying how to run your Dash application. For example:

web: gunicorn app:server

10. Initial deployment: Once everything is set up, push your code to Heroku using the following command:

git push heroku main

11. View your application URL: After the deployment is complete, you can find the URL of your deployed Dash application in the Heroku dashboard under your application's settings.
