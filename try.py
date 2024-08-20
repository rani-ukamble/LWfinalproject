import os
from flask import Flask, Request, redirect, request, session, url_for, render_template
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def get_drive_service():
    """Returns an authenticated Google Drive API service instance."""
    creds = None
    if 'credentials' in session:
        creds = Credentials(**session['credentials'])

    # If there are no valid credentials, initiate OAuth 2.0 flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        session['credentials'] = creds_to_dict(creds)

    return build('drive', 'v3', credentials=creds)

def creds_to_dict(creds):
    """Converts credentials to a serializable dictionary."""
    return {'token': creds.token, 'refresh_token': creds.refresh_token, 'token_uri': creds.token_uri,
            'client_id': creds.client_id, 'client_secret': creds.client_secret, 'scopes': creds.scopes}

@app.route('/')
def index():
    return render_template('drive.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    service = get_drive_service()

    # Search files in Google Drive
    results = service.files().list(q=f"name contains '{query}'",
                                   fields="files(id, name)").execute()

    files = results.get('files', [])

    return render_template('drive.html', files=files)

if __name__ == '__main__':
    app.run(debug=True)
