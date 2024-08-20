from distutils.command import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from bs4 import BeautifulSoup
from httplib2 import Credentials
import requests
from flask_mysqldb import MySQL
from flask import Flask, Request, flash, redirect, render_template, request, session, url_for, send_file, jsonify, send_from_directory
from PIL import Image, ImageFilter
import io
from flask_mail import Mail, Message
from gtts import gTTS
import requests
import vonage
import pandas as pd
import os
from dotenv import load_dotenv
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get MySQL credentials from environment variables
mysql_user = os.getenv('MYSQL_USER')
mysql_password = os.getenv('MYSQL_PASSWORD')

app = Flask(__name__)


# Configure MySQL connection settings
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = mysql_user
app.config['MYSQL_PASSWORD'] = mysql_password
app.config['MYSQL_DB'] = 'test'

# Initialize MySQL object
mysql = MySQL(app)



# Set a secret key for session management and flash messages
app.secret_key = os.getenv('SECRET_KEY')

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'syinfotech57@gmail.com'
app.config['MAIL_PASSWORD'] = 'cuoe xjus ljup bizo'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

VONAGE_API_KEY = 'aae37433'
VONAGE_API_SECRET = 'cveSkI7nwxoQS8JA'

client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
sms = vonage.Sms(client)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PLOTS_FOLDER'] = 'static/plots/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['PLOTS_FOLDER']):
    os.makedirs(app.config['PLOTS_FOLDER'])


mail = Mail(app)


# LinkedIn
# Fetch sensitive information from environment variables
app.secret_key = os.getenv('SECRET_KEY')  # Replace hardcoded secret key with environment variable

CLIENT_ID = os.getenv('CLIENT_ID')  # LinkedIn Client ID from environment variable
CLIENT_SECRET = os.getenv('CLIENT_SECRET')  # LinkedIn Client Secret from environment variable
REDIRECT_URI = os.getenv('REDIRECT_URI')  # LinkedIn Redirect URI from environment variable
AUTH_URL = 'https://www.linkedin.com/oauth/v2/authorization'
TOKEN_URL = 'https://www.linkedin.com/oauth/v2/accessToken'
SEARCH_API_URL = 'https://api.linkedin.com/v2/search?q=people'




SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

# ********************************************************

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_email')
def send_email():
    return render_template('email.html')

@app.route('/send', methods=['POST'])
def email():
    recipient = request.form['email']
    subject = request.form['subject']
    body = request.form['body']
    
    msg = Message(subject, 
                  sender=app.config['MAIL_USERNAME'], 
                  recipients=[recipient])
    msg.body = body
    
    # Handle file upload
    if 'attachment' in request.files:
        attachment = request.files['attachment']
        if attachment.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], attachment.filename)
            attachment.save(file_path)
            with app.open_resource(file_path) as attached_file:
                msg.attach(attachment.filename, attachment.content_type, attached_file.read())
   
    #     return f"Failed to send email. Error: {e}"
    try:
        mail.send(msg)
        flash("Message sent successfully!", "success")  # Flash success message
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Failed to send email. Error: {e}", "error")
        return redirect(url_for('send_email'))

    
# ********************************************************

@app.route('/sms')
def sms_page():
    return render_template('sms.html')

@app.route('/send_sms', methods=['POST'])
def send_sms():
    to_phone_number = request.form['phone']
    message_text = request.form['message']
    
    try:
        response_data = sms.send_message(
            {
                "from": "VonageAPI",
                "to": to_phone_number,
                "text": message_text,
            }
        )
        
        if response_data['messages'][0]['status'] == '0':
            flash("Done!", "success")  # Flash success message
        else:
            flash(f"Message failed with error: {response_data['messages'][0]['error-text']}", "error")
        
    except Exception as e:
        flash(f"Failed to send SMS. Error: {e}", "error")
    
    return redirect(url_for('index'))  # Redirect to index page


# ********************************************************

@app.route('/top5')
def top5():
    return render_template("top5.html")

def google_search(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for item in soup.find_all('h3'):
        parent = item.find_parent('a')
        if parent and parent.get('href'):
            title = item.get_text()
            link = parent.get('href')
            results.append((title, link))
            # Stop after getting the top 5 results
            if len(results) >= 5:
                break
    return results


@app.route('/top5results', methods=['GET', 'POST'])
def top5results():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        if not query:
            flash("Please enter a search query.", "error")
        else:
            results = google_search(query)
    return render_template('top5.html', results=results)

# ********************************************************



@app.route('/audio')
def audio():
    return render_template('audio.html',  audio_file=None)

@app.route('/convert', methods=['POST'])
def convert():
    text = request.form['text']
    language = 'en'  # You can modify this as needed
    
    # Create a gTTS object
    speech = gTTS(text=text, lang=language, slow=False)
    
    # Save the speech to a file
    audio_file = "static/output.mp3"
    speech.save(audio_file)
    
    # Render the page with the audio file
    return render_template('audio.html', audio_file=audio_file, text=text)




# ********************************************************

@app.route('/bulk_email')
def bulk_email():
    return render_template('bulk_email.html')

@app.route('/bulk', methods=['POST'])
def bulk():
    if request.method == 'POST':
        num_emails = int(request.form['num_emails'])
        email_addresses = [request.form[f'email{i+1}'] for i in range(num_emails)]
        subject = request.form['subject']
        body = request.form['body']

        try:
            with mail.connect() as conn:
                for email in email_addresses:
                    msg = Message(
                        recipients=[email],
                        body=body,
                        subject=subject,
                        sender=app.config['MAIL_USERNAME']
                    )
                    
                    if 'attachment' in request.files:
                        attachment = request.files['attachment']
                        if attachment.filename != '':
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], attachment.filename)
                            attachment.save(file_path)
                            with app.open_resource(file_path) as attached_file:
                                msg.attach(attachment.filename, attachment.content_type, attached_file.read())
                    
                    conn.send(msg)
            
            flash('Emails sent successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Failed to send emails. Error: {e}', 'error')

        return redirect(url_for('index'))


# ********************************************************

def get_location_from_ipapi():
    try:
        response = requests.get('https://ipapi.co/json/', timeout=10)
        data = response.json()
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        city = data.get('city')
        region = data.get('region')
        country = data.get('country_name')

        return latitude, longitude, city, region, country
    except requests.exceptions.RequestException as e:
        print(f"Could not retrieve location data: {e}")
        return None, None, None, None, None

@app.route('/location')
def location():
    lat, lon, city, region, country = get_location_from_ipapi()
    if lat and lon:
        return render_template('location.html', latitude=lat, longitude=lon, city=city, region=region, country=country)
    else:
        return "Failed to get location data."




# ********************************************************
# ********************************************************

@app.route('/eda')
def eda():
    return render_template('data_processing.html')


@app.route('/data_processing', methods=['GET', 'POST'])
def data_processing():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the data
        df = pd.read_csv(file_path)
        summary = df.describe().to_html()

        # Example of generating a plot
        plt.figure(figsize=(10, 6))
        df.hist()
        plot_filename = 'histogram.png'
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return render_template('data_processing.html', summary=summary, plot_url=url_for('static', filename=f'plots/{plot_filename}'))
    
    return render_template('data_processing.html')


# ********************************************************

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    with open('iris_model.pkl', 'rb') as file:
        model = pickle.load(file)

# Class names
    class_names = ['setosa', 'versicolor', 'virginica']
    
    if request.method == 'POST':
        try:
            # Extract features from form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Prepare input for prediction
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)

            # Prepare the response
            result = {
                'class_name': class_names[int(prediction[0])]
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})



# ********************************************************

@app.route('/image')
def image():
    return render_template('face.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        # Get the uploaded image file from the form
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')  # Ensure image is in RGB mode
        img_np = np.array(img)  # Convert to NumPy array

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return jsonify({"error": "No face detected."})

        # Create a copy of the original image to overlay the face on
        overlay_img_np = img_np.copy()

        # Crop the first detected face and resize it to fit in the bottom-right corner
        for (x, y, w, h) in faces:
            face = img_np[y:y+h, x:x+w]
            face_size = min(img_np.shape[0] // 4, img_np.shape[1] // 4)
            face_resized = cv2.resize(face, (face_size, face_size))

            # Determine the position for the face in the bottom-right corner
            bottom_right_x = img_np.shape[1] - face_size
            bottom_right_y = img_np.shape[0] - face_size

            # Overlay the resized face on the bottom-right corner of the copied image
            overlay_img_np[bottom_right_y:bottom_right_y+face_size, bottom_right_x:bottom_right_x+face_size] = face_resized
            break  # Take only the first detected face

        # Convert the processed image with the face overlay to base64 string for displaying
        _, buffer = cv2.imencode('.jpg', overlay_img_np)
        main_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"image": main_image_base64})

    except Exception as e:
        return jsonify({"error": str(e)})


# ********************************************************

@app.route('/filter')
def filter():
    return render_template('filter.html')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    img = Image.open(file.stream).convert('RGB')
    img_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
    img.save(img_path)  # Save the uploaded image
    
    return jsonify({'message': 'Image uploaded successfully.'})

@app.route('/original_image', methods=['GET'])
def original_image():
    try:
        img_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        return send_file(img_path, mimetype='image/png')
    except FileNotFoundError:
        return jsonify({'error': 'No image file found.'})

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    filter_name = request.json.get('filter')
    if filter_name not in ['BLUR', 'CONTOUR', 'DETAIL', 'EDGE_ENHANCE']:
        return jsonify({'error': 'Invalid filter.'})

    try:
        img_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        img = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        return jsonify({'error': 'No image file found.'})

    filters = {
        'BLUR': ImageFilter.BLUR,
        'CONTOUR': ImageFilter.CONTOUR,
        'DETAIL': ImageFilter.DETAIL,
        'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE
    }

    img = img.filter(filters[filter_name])
    
    # Save filtered image to buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return send_file(img_bytes, mimetype='image/png', as_attachment=False, download_name='filtered_image.png')


# ********************************************************

@app.route('/numpy_image')
def numpy_image():
    return render_template('numpy_image.html')

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        width = int(request.form['width'])
        height = int(request.form['height'])
        color = request.form['color']

        # Create a custom image using NumPy
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Set the color
        if color.startswith('#'):
            color = color.lstrip('#')
            color = [int(color[i:i+2], 16) for i in (0, 2, 4)]
        else:
            color = [int(c) for c in color.split(',')]

        img_array[:] = color

        # Convert to PIL Image
        img = Image.fromarray(img_array)

        # Save the image to a BytesIO object
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return send_file(img_bytes, mimetype='image/png', as_attachment=False, download_name='custom_image.png')

    except Exception as e:
        return f'Error: {str(e)}'



# ********************************************************
# Load filter images
def load_filter_image(filename):
    img = cv2.imread(os.path.join('static', filename), -1)
    if img is None:
        raise FileNotFoundError(f"Filter image '{filename}' not found.")
    return img

@app.route('/snapfilter')
def snapfilter():
    return render_template('snapfilter.html')

@app.route('/apply_coolfilter', methods=['POST'])
def apply_coolfilter():
    # Load the pre-trained Haar Cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    filters = {
        'sunglass': load_filter_image('sunglass.png'),
        'king': load_filter_image('king.png'),
        'stars': load_filter_image('stars.png'),
        'crown': load_filter_image('crown.png'),
        'flower': load_filter_image('flower.png'),
        'flowers': load_filter_image('flowers.png')

    }
    try:
        # Get the uploaded image file from the form
        file = request.files['image']
        filter_name = request.form.get('filter')

        if filter_name not in filters:
            return jsonify({"error": "Invalid filter."})

        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return jsonify({"error": "No face detected."})

        # Apply the selected filter
        for (x, y, w, h) in faces:
            if filter_name == 'sunglass':
                # Detect eyes within the face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    # Sort eyes by x position to ensure left and right eye order
                    eyes = sorted(eyes, key=lambda ex: ex[0])

                    # Get the eye positions
                    ex1, ey1, ew1, eh1 = eyes[0]  # Left eye
                    ex2, ey2, ew2, eh2 = eyes[1]  # Right eye

                    # Calculate the position for the sunglasses
                    eye_center_x = x + (ex1 + ex2 + ew1 + ew2) // 4
                    eye_center_y = y + (ey1 + ey2 + eh1 + eh2) // 4
                    sunglasses_width = int(1.5 * (ex2 + ew2 - ex1))  # Width slightly larger than distance between eyes
                    sunglasses_height = int(sunglasses_width * filters['sunglass'].shape[0] / filters['sunglass'].shape[1])

                    resized_sunglasses = cv2.resize(filters['sunglass'], (sunglasses_width, sunglasses_height))
                    y_offset = eye_center_y - sunglasses_height // 2
                    x_offset = eye_center_x - sunglasses_width // 2

                    for i in range(resized_sunglasses.shape[0]):
                        for j in range(resized_sunglasses.shape[1]):
                            if resized_sunglasses[i, j][3] != 0:
                                img_np[y_offset + i, x_offset + j] = resized_sunglasses[i, j][:3]

            elif filter_name in ['crown', 'flower', 'king', 'stars','flowers']:
                # Resize filter to the width of the face and position it above the head
                filter_img = filters[filter_name]
                filter_img = cv2.resize(filter_img, (w, int(w * filter_img.shape[0] / filter_img.shape[1])))
                y_offset = y - filter_img.shape[0]
                if y_offset < 0: y_offset = 0

                for i in range(filter_img.shape[0]):
                    for j in range(filter_img.shape[1]):
                        if filter_img[i, j][3] != 0:
                            img_np[y_offset + i, x + j] = filter_img[i, j][:3]

            break  # Apply filter to the first detected face only

        # Convert the processed image to base64 string for displaying
        _, buffer = cv2.imencode('.jpg', img_np)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"image": image_base64})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)})


# ********************************************************

@app.route('/database')
def database():
    return render_template('db.html')

@app.route('/db', methods=['GET', 'POST'])
def db():
    data = None
    error = None
    if request.method == 'POST':
        query = request.form['query']
        try:
            cur = mysql.connection.cursor()
            cur.execute(query)
            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                mysql.connection.commit()
                data = "Query executed successfully."
            else:
                data = cur.fetchall()
            cur.close()
        except Exception as e:
            error = str(e)

    return render_template('db.html', data=data, error=error)


# ************************************************************
@app.route('/linkedin')
def linkedin():
    return render_template('linkedin.html')

@app.route('/login')
def login():
    auth_link = f"{AUTH_URL}?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=r_liteprofile%20r_emailaddress"
    return redirect(auth_link)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }
    response = requests.post(TOKEN_URL, data=data)
    access_token = response.json().get('access_token')
    session['access_token'] = access_token
    return redirect(url_for('search'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'access_token' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        query = request.form['query']
        headers = {'Authorization': f"Bearer {session['access_token']}"}
        search_response = requests.get(f"{SEARCH_API_URL}&keywords={query}", headers=headers)
        results = search_response.json()
        return render_template('linkedin.html', results=results)
    
    return render_template('linkedin.html')

# ****************************************************************

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

@app.route('/drive')
def drive():
    return render_template('drive.html')

@app.route('/searchdrive', methods=['POST'])
def searchdrive():
    query = request.form['query']
    service = get_drive_service()

    # Search files in Google Drive
    results = service.files().list(q=f"name contains '{query}'",
                                   fields="files(id, name)").execute()

    files = results.get('files', [])

    return render_template('drive.html', files=files)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)






