from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import re
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import shutil
import time
from google.cloud import aiplatform
import numpy as np
import os
import requests
from PIL import Image
import visuallayer as vl
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim

import json

app = Flask(__name__)
app.secret_key = 'Chiccopicco1!'


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


aiplatform.init(project='wildlifeweb', location='europe-west2')

model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_s/classification/2"
model = hub.load(model_url)

with open('imagenet22k.txt') as f:
    class_labels = json.load(f)


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) 
    image = image / 255.0
    image = image[np.newaxis, ...]
    return image


def predict_image_class(image_path):
    image = load_and_preprocess_image(image_path)
    predictions = model(image)
    predicted_class_index = np.argmax(predictions, axis=-1)
    predicted_label = class_labels[str(predicted_class_index[0])]
    return predicted_label


@app.route('/upload', methods=['GET', 'POST'])
def upload_and_classify_image():

    if 'username' not in session and 'image' in request.files:
        return render_template('upload.html', message="Please login to upload image")

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and not allowed_file(file.filename):
            return render_template('upload.html', message2="Invalid file type. Pictures must be a JPG, JPEG, PNG, WEBP or GIF.")
        
        if file and file.filename:
            filename = secure_filename(file.filename)
            image_path = os.path.join('static', 'uploads', filename)
            file.save(image_path)
            predicted_label = predict_image_class(image_path)
            print(predicted_label)
            common_name, scientific_name = parse_label(predicted_label)
            species_description = fetch_species_description(predicted_label)
            print(common_name)
            print(scientific_name)
            print(species_description)
            temp_image_url = url_for('static', filename=image_path[7:])  
            
            if common_name==scientific_name:
                return render_template('upload.html', temp_image_url=temp_image_url, prediction=predicted_label, temp_image_path=image_path, common_name=common_name,species_description=species_description, predicted_label=predicted_label)

            else:
                return render_template('upload.html', temp_image_url=temp_image_url, prediction=predicted_label, temp_image_path=image_path, common_name=common_name, scientific_name=scientific_name,species_description=species_description, predicted_label=predicted_label)

    return render_template('upload.html', prediction=None)


@app.route('/confirm_upload', methods=['POST'])
def confirm_upload():
    image_path = request.form['image_path']
    predicted_label = request.form['predicted_label']  
    description = request.form.get('description', '')
    
    latitude = request.form.get('latitude', None)
    longitude = request.form.get('longitude', None)
    latitude = float(latitude) if latitude else 0.0  
    longitude = float(longitude) if longitude else 0.0  
    location_name = request.form.get('locationName', 'Unknown Location')  
    
    common_name, scientific_name = parse_label(predicted_label)
    
    species_description = request.form.get('species_description', 'Default description')

    username = session.get('username')
    user_id = getUserID(username)

    city, country = get_location_details(latitude, longitude)

    with mysql.connection.cursor() as cursor:
        cursor.execute("SELECT SpeciesID FROM species WHERE ScientificName = %s", (scientific_name,))
        species = cursor.fetchone()

        if species:
            species_id = species['SpeciesID']
        else:
            cursor.execute("INSERT INTO species (CommonName, ScientificName, Description, PredictedLabel) VALUES (%s, %s, %s, %s)", 
                           (common_name, scientific_name, species_description, predicted_label))
            species_id = cursor.lastrowid
            mysql.connection.commit()
            
        cursor.execute("SELECT LocationID FROM Locations WHERE Latitude = %s AND Longitude = %s AND LocationName = %s", (latitude, longitude, location_name))
        location = cursor.fetchone()
        if location:
            location_id = location['LocationID']
        else:
            cursor.execute("INSERT INTO Locations (Latitude, Longitude, LocationName, City, Country) VALUES (%s, %s, %s, %s, %s)", 
                           (latitude, longitude, location_name, city, country))
            location_id = cursor.lastrowid
            mysql.connection.commit()
        
        confirmed_image_path = os.path.join('static', 'confirmed_uploads', os.path.basename(image_path))
        shutil.move(image_path, confirmed_image_path)
        
        sql = """
            INSERT INTO Sightings (UserID, SpeciesID, ImagePath, Description, LocationID) 
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (user_id, species_id, confirmed_image_path, description, location_id))
        sighting_id = cursor.lastrowid
        mysql.connection.commit()

    flash('Sighting uploaded and confirmed successfully!', 'success')
    return render_template('congratulations.html', 
                           image_path=confirmed_image_path, 
                           common_name=common_name, 
                           scientific_name=scientific_name, 
                           description=description,
                           sighting_id=sighting_id)



def get_location_details(latitude, longitude):
    city = 'Unknown'
    country = 'Unknown'
    if latitude != 0 and longitude != 0:
        result = gmaps.reverse_geocode((latitude, longitude))
        if result:
            address = result[0].get('address_components', [])
            city = next((item['long_name'] for item in address if 'locality' in item['types']), 'Unknown')
            if city == 'Unknown':  
                city = next((item['long_name'] for item in address if 'administrative_area_level_2' in item['types']), 'Unknown')
            country = next((item['long_name'] for item in address if 'country' in item['types']), 'Unknown')
    return city, country




def parse_label(label):
    """Parse the label to extract the common and scientific names."""
    names = label.split(', ')  
    if not names:
        return None, None  
    common_name = names[0].replace('_', ' ').title()  
    scientific_name = names[-1].replace('_', ' ').title()
    return common_name, scientific_name





def fetch_species_description(name_variations_str):
    url = "https://en.wikipedia.org/w/api.php"
    names = [name.strip() for name in name_variations_str.split(',')]
    last_variation = names[-1]
    search_queue = [last_variation]     
    if '_' in last_variation:
        search_queue.extend(last_variation.split('_'))
    search_queue.extend(names[:-1])
    
    for name in search_queue:
        formatted_name = name.replace(" ", "_")
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": formatted_name,
            "format": "json"
        }
        search_response = requests.get(url, params=search_params)
        search_data = search_response.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            continue 
        first_result_title = search_results[0].get("title", "")

        content_params = {
            "action": "query",
            "format": "json",
            "titles": first_result_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }
        
        content_response = requests.get(url, params=content_params)
        content_data = content_response.json()
        pages = content_data.get("query", {}).get("pages", {})
        for page_id in pages:
            description = pages[page_id].get("extract", "Description not available.")
            return description  
    return "Description not found."  




import requests

def fetch_relevant_species_images(name):
    base_url = "https://en.wikipedia.org/w/api.php"
    formatted_name = name.replace(" ", "_")
    
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": formatted_name
    }
    
    try:
        search_response = requests.get(base_url, params=search_params)
        search_response.raise_for_status()  
        search_data = search_response.json()
        search_results = search_data.get("query", {}).get("search", [])
        
        if not search_results:
            return []  
        
        page_id = search_results[0]["pageid"]

        image_params = {
            "action": "query",
            "format": "json",
            "pageids": page_id,
            "prop": "images"
        }
        image_response = requests.get(base_url, params=image_params)
        image_response.raise_for_status()
        image_data = image_response.json()
        images = image_data.get("query", {}).get("pages", {}).get(str(page_id), {}).get("images", [])
        
        relevant_images = []
        species_keywords = [formatted_name.lower(), name.lower().split()[-1]]
        for image in images:
            if any(keyword in image['title'].lower() for keyword in species_keywords):
                if image['title'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    relevant_images.append(image['title'])

        if not relevant_images:
            return []  

        image_info_params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": "|".join(relevant_images)
        }
        image_info_response = requests.get(base_url, params=image_info_params)
        image_info_response.raise_for_status()
        image_info_data = image_info_response.json()

        final_images = []
        pages = image_info_data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "imageinfo" in page:
                url = page["imageinfo"][0]["url"]
                alt_text = page.get("title", "No description available").split(':')[1]  
                final_images.append({'url': url, 'alt': alt_text})

        return final_images
    except requests.RequestException as e:
        print(f"Error fetching images: {e}")
        return []  








def getUserID(username):
    try:
        with mysql.connection.cursor() as cursor:
            cursor.execute("SELECT UserID FROM Users WHERE Username = %s", (username,))
            user = cursor.fetchone()
            if user:
                return user['UserID']
            else:
                return None  
    except Exception as e:
        print(f"An error occurred while trying to fetch UserID for username {username}: {e}")
        return None  
    
    
# @app.route('/sightings_feed')
# def sightings_feed():
#     with mysql.connection.cursor() as cursor:
#         sql = """
#         SELECT s.SightingID, u.Username, 
#             sp.CommonName, sp.ScientificName, 
#             s.ImagePath, s.Timestamp, s.Description,
#             l.Latitude, l.Longitude, l.LocationName # Ensure these fields are selected
#         FROM Sightings s
#         JOIN Users u ON s.UserID = u.UserID
#         JOIN Species sp ON s.SpeciesID = sp.SpeciesID
#         JOIN Locations l ON s.LocationID = l.LocationID  # Assuming this join is correct
#         ORDER BY s.Timestamp DESC
#         """
#         cursor.execute(sql)
#         sightings = cursor.fetchall()

#         # Print latitude and longitude for each sighting
#         for sighting in sightings:
#             print(sighting['Latitude'], sighting['Longitude'])
#             print("hi")
#     # Pass sightings to the template
#     return render_template('discover.html', sightings=sightings)




app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Chiccopicco1!' 
app.config['MYSQL_DB'] = 'WildlifeWeb'  #
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)
bcrypt = Bcrypt(app)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')



@app.route('/discover_gallery')
def discover_gallery():
    season = request.args.get('season', 'all')
    with mysql.connection.cursor() as cursor:
        query = """
        SELECT s.ImagePath, s.SightingID, s.Timestamp, sp.CommonName, sp.ScientificName,
               COUNT(l.SightingID) AS LikeCount
        FROM Sightings s
        JOIN Species sp ON s.SpeciesID = sp.SpeciesID
        LEFT JOIN Likes l ON s.SightingID = l.SightingID
        WHERE s.ImagePath IS NOT NULL
        """

        if season != 'all':
            if season == 'Spring':
                query += " AND MONTH(s.Timestamp) IN (3, 4, 5)"
            elif season == 'Summer':
                query += " AND MONTH(s.Timestamp) IN (6, 7, 8)"
            elif season == 'Autumn':
                query += " AND MONTH(s.Timestamp) IN (9, 10, 11)"
            elif season == 'Winter':
                query += " AND MONTH(s.Timestamp) IN (12, 1, 2)"

        query += " GROUP BY s.SightingID ORDER BY s.Timestamp DESC"
        cursor.execute(query)
        results = cursor.fetchall()

    images = [{
        'url': url_for('static', filename='confirmed_uploads/' + os.path.basename(result['ImagePath'])),
        'alt': result['CommonName'],
        'name': result['CommonName'],
        'scientific_name': result['ScientificName'],
        'sighting_id': result['SightingID'],
        'date_added': result['Timestamp'].strftime("%Y-%m-%d"),
        'likes': result['LikeCount']
    } for result in results if os.path.exists(os.path.join(app.static_folder, 'confirmed_uploads', os.path.basename(result['ImagePath'])))]

    return render_template('discover_gallery.html', images=images, season=season)


@app.route('/image_details')
def image_details():
    image_path = request.args.get('image_path')
    image_path = image_path[1:]  
    if not image_path:
        return "Image not specified", 400  

    query = """
        SELECT s.SightingID, s.Timestamp, s.Description, s.ImagePath, l.LocationName, l.latitude, l.longitude, l.City, l.Country, sp.CommonName, sp.ScientificName, u.Username
        FROM Sightings s
        JOIN Locations l ON s.LocationID = l.LocationID
        JOIN Species sp ON s.SpeciesID = sp.SpeciesID
        JOIN Users u ON s.UserID = u.UserID
        WHERE s.ImagePath = %s
    """
    with mysql.connection.cursor() as cursor:
        cursor.execute(query, [image_path])
        sighting = cursor.fetchone()

    if not sighting:
        return "Sighting not found", 404

    return render_template('sighting_detail.html', sighting=sighting)




@app.route('/sightings_feed')
def sightings_feed():
    username = session.get('username') 
    with mysql.connection.cursor() as cursor:
        sql = """
        SELECT s.SightingID, u.Username, u.ProfilePicture, sp.CommonName, sp.SpeciesID, sp.ScientificName, s.ImagePath, s.Timestamp, s.Description,
            l.Latitude, l.Longitude, COUNT(lk.UserID) AS like_count
        FROM Sightings s
        JOIN Users u ON s.UserID = u.UserID
        JOIN Species sp ON s.SpeciesID = sp.SpeciesID
        JOIN Locations l ON s.LocationID = l.LocationID
        LEFT JOIN Likes lk ON s.SightingID = lk.SightingID
        GROUP BY s.SightingID
        ORDER BY s.Timestamp DESC
        """
        cursor.execute(sql)
        sightings = cursor.fetchall()

        user_id = None
        if username:
            cursor.execute('SELECT UserID FROM Users WHERE Username = %s', (username,))
            user_result = cursor.fetchone()
            if user_result:
                user_id = user_result['UserID']

        for sighting in sightings:
            if user_id:
                cursor.execute('SELECT * FROM Likes WHERE UserID = %s AND SightingID = %s', (user_id, sighting['SightingID']))
                sighting['liked'] = cursor.fetchone() is not None
            else:
                sighting['liked'] = False

    return render_template('discover.html', sightings=sightings)



@app.route('/walk')
def walk():
    cursor = mysql.connection.cursor()
    query = "SELECT Path FROM Walk WHERE Name = 'Deer Walk'"
    cursor.execute(query)
    route_data = cursor.fetchone()
    if route_data:
        route_json = route_data 
    else:
        route_json = '[]'  
    return render_template('walk.html', route_json=route_json)


@app.route('/account')
def account():
    if 'logged_in' in session and session['logged_in']:
        username = session.get('username')

        with mysql.connection.cursor() as cursor:
            cursor.execute("SELECT FirstName, Username, Surname, Email, ProfilePicture FROM Users WHERE Username = %s", [username])
            user_details = cursor.fetchone()

        if user_details:
            return render_template('account.html', user_details=user_details)
        else:
            flash("User details not found.", "error")
            return redirect(url_for('login'))
    else:
        flash("Please log in to view this page.", "warning")
        return redirect(url_for('login'))

    
    
@app.route('/discover/recent-sightings')
def discover_recent_sightings():
    return render_template('recent-sightings.html', show_discover_tabs=True)



@app.route('/discover/common-species')
def discover_common_species():
    return render_template('common-species.html', show_discover_tabs=True)




@app.route('/api/sightings')
def sightings():

    sightings = [
   
    {'lat': 51.451902, 'lng': -2.600750, 'description': 'Bristol Cathedral'},
    {'lat': 51.454514, 'lng': -2.627645, 'description': 'Clifton Suspension Bridge'},
    {'lat': 51.446590, 'lng': -2.618107, 'description': 'SS Great Britain'},
    {'lat': 51.470747, 'lng': -2.622179, 'description': 'Bristol Zoo Gardens'},
    {'lat': 51.432676, 'lng': -2.661629, 'description': 'Ashton Court Estate'},
    {'lat': 51.456310, 'lng': -2.607016, 'description': 'Bristol Museum & Art Gallery'},
    {'lat': 51.455235, 'lng': -2.604333, 'description': 'Cabot Tower'},
    {'lat': 51.448240, 'lng': -2.598200, 'description': 'Arnolfini'},
    {'lat': 51.468489, 'lng': -2.635743, 'description': 'The Downs'},
    {'lat': 51.447935, 'lng': -2.596065, 'description': 'M Shed'}
]

    
    return jsonify(sightings)




@app.route('/api/walk-routes')
def walk_routes():

    walk_routes = [
        {
            "name": "Route 1",
            "path": [
                {"lat": 51.4545, "lng": -2.5879},
                {"lat": 51.4555, "lng": -2.5870},
            ]
        },
    ]
    return jsonify(walk_routes)


@app.route('/logout')
def logout():
    session.clear()

    return redirect(url_for('login'))


def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'\d', password):  
        return False
    if not re.search(r'[A-Z]', password):  
        return False
    return True

from geopy.geocoders import Nominatim
import googlemaps
import plotly.express as px
import plotly.io as pio
import pandas as pd

gmaps = googlemaps.Client(key='AIzaSyCC1vhYZf5T6VR7JNAJNuda1LgjeP7VSw4')

@app.route('/admin')
def admin_dashboard():
    cursor = mysql.connection.cursor()

    cursor.execute("SELECT COUNT(*) AS total_users FROM Users")
    total_users = cursor.fetchone()['total_users']

    cursor.execute("SELECT COUNT(*) AS active_users FROM Users WHERE last_login > DATE_SUB(NOW(), INTERVAL 1 DAY)")
    active_users = cursor.fetchone()['active_users']

    cursor.execute("SELECT COUNT(*) AS new_registrations FROM Users WHERE registration_date > DATE_SUB(NOW(), INTERVAL 1 WEEK)")
    new_registrations = cursor.fetchone()['new_registrations']

    cursor.execute("SELECT COUNT(*) AS total_species FROM Species")
    total_species = cursor.fetchone()['total_species']

    cursor.execute("SELECT COUNT(*) AS total_sightings FROM Sightings")
    total_sightings = cursor.fetchone()['total_sightings']

    cursor.execute("SELECT LocationID, Latitude, Longitude FROM Locations")
    locations = cursor.fetchall()

    cities = {}
    countries = set()

    for location in locations:
        location_id = location['LocationID']
        latitude = location['Latitude']
        longitude = location['Longitude']
        if latitude != 0 and longitude != 0:
            result = gmaps.reverse_geocode((latitude, longitude))
            if result:
                address = result[0].get('address_components', [])
                city = next((item['long_name'] for item in address if 'locality' in item['types']), '')
                if not city:  
                    city = next((item['long_name'] for item in address if 'administrative_area_level_2' in item['types']), '')

                country = next((item['long_name'] for item in address if 'country' in item['types']), '')

                if city:
                    cities[location_id] = city
                if country:
                    countries.add(country)

    total_cities = len(set(cities.values()))
    total_countries = len(countries)

    sightings_data = []
    for location_id, city in cities.items():
        cursor.execute("""
            SELECT Species.CommonName AS species, COUNT(Sightings.SightingID) AS count
            FROM Sightings
            JOIN Species ON Sightings.SpeciesID = Species.SpeciesID
            WHERE Sightings.LocationID = %s
            GROUP BY Species.CommonName
        """, (location_id,))
        city_sightings = cursor.fetchall()
        for sighting in city_sightings:
            sightings_data.append({
                'species': sighting['species'],
                'city': city,
                'count': sighting['count']
            })

    df = pd.DataFrame(sightings_data)

    fig = px.bar(df, x='city', y='count', color='species', title='Species Sightings per City')

    graph_html = pio.to_html(fig, full_html=False)

    stats = {
        'total_users': total_users,
        'active_users': active_users,
        'new_registrations': new_registrations,
        'total_species': total_species,
        'total_sightings': total_sightings,
        'total_cities': total_cities,
        'total_countries': total_countries
    }

    return render_template('admin_dashboard.html', stats=stats, graph_html=graph_html)

@app.route('/admin/users')
def admin_users():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM Users")
    users = cursor.fetchall()
    return render_template('admin_users.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    cursor = mysql.connection.cursor()

    cursor.execute("DELETE FROM Likes WHERE UserID = %s", (user_id,))
    
    cursor.execute("DELETE FROM Sightings WHERE UserID = %s", (user_id,))
    
    cursor.execute("DELETE FROM Users WHERE UserID = %s", (user_id,))

    mysql.connection.commit()
    return redirect(url_for('admin_users'))



@app.route('/admin/sightings')
def admin_sightings():
    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT s.SightingID, u.Username, sp.CommonName, s.Timestamp, s.Description,
               l.Latitude, l.Longitude
        FROM Sightings s
        JOIN Users u ON s.UserID = u.UserID
        JOIN Species sp ON s.SpeciesID = sp.SpeciesID
        JOIN Locations l ON s.LocationID = l.LocationID
        ORDER BY s.Timestamp DESC
    """)
    sightings = cursor.fetchall()

    for sighting in sightings:
        latitude = sighting['Latitude']
        longitude = sighting['Longitude']
        if latitude != 0 and longitude != 0:
            result = gmaps.reverse_geocode((latitude, longitude))
            if result:
                address = result[0].get('address_components', [])
                city = next((item['long_name'] for item in address if 'locality' in item['types']), '')
                if not city:  
                    city = next((item['long_name'] for item in address if 'administrative_area_level_2' in item['types']), '')
                sighting['City'] = city
            else:
                sighting['City'] = 'Unknown'
        else:
            sighting['City'] = 'Unknown'

    return render_template('admin_sightings.html', sightings=sightings)



@app.route('/admin/delete_sighting/<int:sighting_id>', methods=['POST'])
def delete_sighting(sighting_id):
    cursor = mysql.connection.cursor()
    print(sighting_id)

    sql = "SELECT SpeciesID FROM Sightings WHERE SightingID = %s"
    cursor.execute(sql, (sighting_id,))
    result = cursor.fetchone()

    if result:
        species_id = result[0] if isinstance(result, tuple) else result['SpeciesID']

        cursor.execute("DELETE FROM Likes WHERE SightingID = %s", (sighting_id,))

        cursor.execute("DELETE FROM Sightings WHERE SightingID = %s", (sighting_id,))

        count_sql = "SELECT COUNT(*) FROM Sightings WHERE SpeciesID = %s"
        cursor.execute(count_sql, (species_id,))
        count_result = cursor.fetchone()
        count = count_result[0] if isinstance(count_result, tuple) else count_result['COUNT(*)']

        if count == 0:
            try:
                cursor.execute("DELETE FROM Species WHERE SpeciesID = %s", (species_id,))
            except mysql.connection.IntegrityError as e:
                print("Cannot delete species as it is still referenced by sightings.", e)
    
    mysql.connection.commit()

    return redirect(url_for('admin_sightings'))


from flask import request, render_template, send_file
from fpdf import FPDF

@app.route('/admin/reports', methods=['GET', 'POST'])
def admin_reports():
    error = request.args.get('error')  

    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        report_type = request.form.get('report_type')
        
        return generate_pdf_report(report_type, start_date, end_date)

    return render_template('admin_reports.html', error=error)












from fpdf import FPDF
from flask import send_file, current_app


def generate_pdf_report(report_type, start_date, end_date):
    cursor = mysql.connection.cursor()  

    pdf = FPDF(orientation='L')  
    pdf.add_page()
    pdf.set_font("Arial", size=8) 

    pdf.cell(0, 10, f"Report Type: {report_type.capitalize()} from {start_date} to {end_date}", ln=1, align='C')
    pdf.ln(4)  

    headers = []
    query = ""
    col_widths = {}
    total_label = ""
    if report_type == 'sightings':
        headers = ["SightingID", "Username", "CommonName", "ScientificName", "Timestamp", "Description", "City", "Country"]
        col_widths = {
            "SightingID": 20,
            "Username": 30,
            "CommonName": 35,
            "ScientificName": 45,
            "Timestamp": 40,
            "Description": 50,
            "City": 30,
            "Country": 30
        }
        query = f"""
            SELECT s.SightingID, u.Username, sp.CommonName, sp.ScientificName, s.Timestamp, s.Description,
                   l.City, l.Country
            FROM Sightings s
            JOIN Users u ON s.UserID = u.UserID
            JOIN Species sp ON s.SpeciesID = sp.SpeciesID
            JOIN Locations l ON s.LocationID = l.LocationID
            WHERE s.Timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY s.Timestamp DESC
        """
        total_label = "Total Sightings"
    
    elif report_type == 'species':
        headers = ["SpeciesID", "CommonName", "ScientificName", "City", "Country"]
        col_widths = {
            "SpeciesID": 20,
            "CommonName": 45,
            "ScientificName": 45,
            "City": 30,
            "Country": 30
        }
        query = f"""
            SELECT sp.SpeciesID, sp.CommonName, sp.ScientificName, l.City, l.Country
            FROM Species sp
            JOIN Sightings s ON sp.SpeciesID = s.SpeciesID
            JOIN Locations l ON s.LocationID = l.LocationID
            WHERE s.Timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY sp.SpeciesID
        """
        total_label = "Total Species"
    
    elif report_type == 'users':
        headers = ["UserID", "FirstName", "Surname", "Email", "Registration_Date"]
        col_widths = {
            "UserID": 20,
            "FirstName": 30,
            "Surname": 30,
            "Email": 45,
            "Registration_Date": 45
        }
        query = f"""
            SELECT UserID, FirstName, Surname, Email, Registration_Date 
            FROM Users 
            WHERE Registration_Date BETWEEN '{start_date}' AND '{end_date}'
        """
        total_label = "Total Users"
    
    elif report_type == 'locations':
        headers = ["LocationID", "LocationName", "Latitude", "Longitude", "City", "Country"]
        col_widths = {
            "LocationID": 20,
            "LocationName": 45,
            "Latitude": 30,
            "Longitude": 30,
            "City": 30,
            "Country": 30
        }
        query = "SELECT LocationID, LocationName, Latitude, Longitude, City, Country FROM Locations"
        total_label = "Total Locations"

    cursor.execute(query)
    rows = cursor.fetchall()
    total_count = len(rows)  

    if not rows:
        flash('No data available for the selected range or type.', 'warning') 
        return redirect(url_for('admin_reports', error="No data is available for the selected date range."))
        
    pdf.cell(0, 10, f"{total_label}: {total_count}", ln=1, align='C')
    pdf.ln(4)  

    for header in headers:
        pdf.cell(col_widths.get(header, 20), 10, header, border=1)
    pdf.ln()

    for row in rows:
        for header in headers:
            pdf.cell(col_widths.get(header, 20), 10, str(row[header]), border=1)
        pdf.ln()

    pdf_output = "/tmp/report.pdf"  
    pdf.output(pdf_output)
    return send_file(pdf_output, as_attachment=True)




    
@app.route('/admin_logout')
def admin_logout():
    session.clear()
    
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        
        admin_username = 'admin'
        admin_password = 'admin123'
        
        username = request.form['username']
        password_candidate = request.form['password']
        cursor = mysql.connection.cursor()
        result = cursor.execute("SELECT * FROM Users WHERE Username = %s", [username])
        if username == admin_username and password_candidate == admin_password:
            session['logged_in'] = True
            session['username'] = username
            session['is_admin'] = True  
            return redirect(url_for('admin_dashboard'))
        if result > 0:
            data = cursor.fetchone()
            print(data)  
            password = data['PasswordHash']
            if bcrypt.check_password_hash(password, password_candidate):
                cursor.execute("UPDATE Users SET Last_Login = NOW() WHERE Username = %s", [username])
                mysql.connection.commit()
                session['logged_in'] = True
                session['username'] = username
                flash('You are now logged in', 'success')
                return redirect(url_for('home')) 
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)
    return render_template('login.html')
    
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        
        if not is_valid_email(email):
            return render_template("register.html", error="Email is not in a valid format.")
        
        
        with mysql.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM Users WHERE Username = %s", (username,))
            if cursor.fetchone():
                return render_template("register.html", error="That username is already taken.")
            cursor.execute("SELECT * FROM Users WHERE Email = %s", (email,))
            if cursor.fetchone():
                return render_template("register.html", error="Email already in use.")

        if not is_valid_password(password):
            return render_template("register.html", error="Your password must be at least 8 characters long, include a digit and an uppercase letter.")

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        try:
            with mysql.connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO Users (Username, Email, PasswordHash, FirstName, Surname, Registration_Date, Last_Login) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())",
                    (username, email, hashed_password, first_name, last_name)
                )
                mysql.connection.commit()
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('account'))  
        except Exception as e:
            mysql.connection.rollback()
            return render_template("register.html", error=f"Error during registration: {e}")

    return render_template("register.html")


def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$'
    return re.match(email_regex, email)



@app.route('/get_address', methods=['POST'])
def get_address():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']

    API_KEY = 'AIzaSyCC1vhYZf5T6VR7JNAJNuda1LgjeP7VSw4'

    response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={API_KEY}')
    address_info = response.json()
    print(response.json())

    if address_info['status'] == 'OK':
        print(address_info)
        address = address_info['results'][0]['formatted_address']
        return jsonify({'address': address})
    else:
        return jsonify({'error': 'Address not found'}), 400
    
    
    
    
    
    
    
@app.route('/map_view_post')
def map_view_post():

    focus = request.args.get('focus', default=None, type=int)
    return render_template('map_view.html', focus=focus, sightings=sightings)



    
@app.route('/map_view')
def map_view():
    focus_id = request.args.get('focus_id', default=None, type=int)

    with mysql.connection.cursor() as cursor:
        sql = """
        SELECT s.SightingID, u.Username,
            sp.CommonName, sp.ScientificName, sp.SpeciesID,
            s.ImagePath, s.Timestamp, s.Description,
            l.Latitude, l.Longitude
        FROM Sightings s
        JOIN Users u ON s.UserID = u.UserID
        JOIN Species sp ON s.SpeciesID = sp.SpeciesID
        JOIN Locations l ON s.LocationID = l.LocationID
        ORDER BY s.Timestamp DESC
        """
        cursor.execute(sql)
        sightings = cursor.fetchall()

    sightings_json = json.dumps(sightings, default=str)  
    return render_template('map_view.html', sightings=sightings, sightings_json=sightings_json, focus_id=focus_id)

    
    
@app.route('/sighting_details/<sighting_id>')
def sighting_details(sighting_id):
    with mysql.connection.cursor(dictionary=True) as cursor:
        cursor.execute("""
            SELECT s.*, l.LocationName, sp.CommonName, sp.ScientificName
            FROM Sightings s
            JOIN Locations l ON s.LocationID = l.LocationID
            JOIN Species sp ON s.SpeciesID = sp.SpeciesID
            WHERE s.SightingID = %s
        """, (sighting_id,))
        sighting = cursor.fetchone()

    if not sighting:
        return "Sighting not found", 404

    return render_template('sighting_detail.html', sighting=sighting)

    
@app.route('/species_view')
def species_view():
    species_id = request.args.get('species_id', type=int)
    username = session.get('username')  

    if species_id is not None:
        with mysql.connection.cursor() as cursor:
            query = """
            SELECT s.SightingID, s.Timestamp, s.Description, s.ImagePath, l.LocationName, sp.CommonName, sp.ScientificName, sp.SpeciesID, u.Username, u.ProfilePicture,
                   COUNT(lk.UserID) AS like_count
            FROM Sightings s
            JOIN Locations l ON s.LocationID = l.LocationID
            JOIN Species sp ON s.SpeciesID = sp.SpeciesID
            JOIN Users u ON s.UserID = u.UserID
            LEFT JOIN Likes lk ON s.SightingID = lk.SightingID
            WHERE s.SpeciesID = %s
            GROUP BY s.SightingID
            ORDER BY s.Timestamp DESC
            """
            cursor.execute(query, (species_id,))
            sightings = cursor.fetchall()

            total_sightings = len(sightings)

            user_id = None
            if username:
                cursor.execute('SELECT UserID FROM Users WHERE Username = %s', (username,))
                user_result = cursor.fetchone()
                if user_result:
                    user_id = user_result['UserID']

            for sighting in sightings:
                if user_id:
                    cursor.execute('SELECT * FROM Likes WHERE UserID = %s AND SightingID = %s', (user_id, sighting['SightingID']))
                    sighting['liked'] = cursor.fetchone() is not None
                else:
                    sighting['liked'] = False

        if sightings:
            return render_template('species_view.html', sightings=sightings, total_sightings=total_sightings, species_name=sightings[0]['CommonName'])
        else:
            return render_template('species_view.html', error="No sightings found for the selected species.")

    return render_template('species_view.html', error="No species selected.")








@app.route('/like_sighting/<sighting_id>', methods=['POST'])
def like_sighting(sighting_id):
    username = session.get('username')  
    if not username:
        return jsonify({'error': 'You must be logged in to like images'}), 403

    cursor = mysql.connection.cursor()
    cursor.execute('SELECT UserID FROM Users WHERE Username = %s', (username,))
    user_result = cursor.fetchone()
    if not user_result:
        return jsonify({'error': 'User not found'}), 404

    user_id = user_result['UserID']  

    cursor.execute('SELECT * FROM Likes WHERE UserID = %s AND SightingID = %s', (user_id, sighting_id))
    if cursor.fetchone():
        return jsonify({'error': 'You have already liked this sighting'}), 409

    cursor.execute('INSERT INTO Likes (UserID, SightingID) VALUES (%s, %s)', (user_id, sighting_id))
    mysql.connection.commit()
    
    cursor.execute('SELECT COUNT(*) AS like_count FROM Likes WHERE SightingID = %s', (sighting_id,))
    like_result = cursor.fetchone()
    like_count = like_result['like_count'] if like_result else 0

    mysql.connection.commit()

    return jsonify({'success': 'Sighting liked successfully', 'like_count': like_count}), 200


@app.route('/unlike_sighting/<sighting_id>', methods=['POST'])
def unlike_sighting(sighting_id):
    username = session.get('username')  
    if not username:
        return jsonify({'error': 'You must be logged in to unlike images'}), 403

    cursor = mysql.connection.cursor()
    cursor.execute('SELECT UserID FROM Users WHERE Username = %s', (username,))
    user_result = cursor.fetchone()
    if not user_result:
        return jsonify({'error': 'User not found'}), 404

    user_id = user_result['UserID']  

    cursor.execute('SELECT * FROM Likes WHERE UserID = %s AND SightingID = %s', (user_id, sighting_id))
    if not cursor.fetchone():
        return jsonify({'error': 'You have not liked this sighting'}), 409

    cursor.execute('DELETE FROM Likes WHERE UserID = %s AND SightingID = %s', (user_id, sighting_id))
    mysql.connection.commit()
    
    cursor.execute('SELECT COUNT(*) AS like_count FROM Likes WHERE SightingID = %s', (sighting_id,))
    like_result = cursor.fetchone()
    like_count = like_result['like_count'] if like_result else 0

    mysql.connection.commit()

    return jsonify({'success': 'Sighting unliked successfully', 'like_count': like_count}), 200



@app.route('/common_species')
def common_species():
    with mysql.connection.cursor() as cursor:
        query = """
        SELECT sp.SpeciesID, sp.CommonName, sp.ScientificName, COUNT(s.SightingID) AS sighting_count
        FROM Species sp
        JOIN Sightings s ON sp.SpeciesID = s.SpeciesID
        GROUP BY sp.SpeciesID
        ORDER BY sighting_count DESC
        LIMIT 10
        """
        cursor.execute(query)
        species_list = cursor.fetchall()

    return render_template('common_species.html', species_list=species_list)


@app.route('/more_info')
def more_info():
    species_id = request.args.get('focus_id', type=int)
    if not species_id:
        return "Species not found", 404

    with mysql.connection.cursor() as cursor:
        query = """
        SELECT sp.CommonName, sp.ScientificName, sp.Description, sp.PredictedLabel
        FROM Species sp
        WHERE sp.SpeciesID = %s
        """
        cursor.execute(query, (species_id,))
        species_info = cursor.fetchone()

        locations_query = """
        SELECT l.Latitude, l.Longitude
        FROM Locations l
        JOIN Sightings s ON l.LocationID = s.LocationID
        WHERE s.SpeciesID = %s
        """
        cursor.execute(locations_query, (species_id,))
        locations = cursor.fetchall()

    if not species_info:
        return "Species not found", 404

    images = fetch_relevant_species_images(species_info['CommonName'])

    return render_template('more_info.html', species=species_info, images=images, locations=locations)


@app.route('/discover_main')
def discover_main():
    return render_template('discover_main.html')


def month_to_season(month):
    if month in (3, 4, 5):
        return 'Spring'
    elif month in (6, 7, 8):
        return 'Summer'
    elif month in (9, 10, 11):
        return 'Autumn'
    else:
        return 'Winter'

app.config['UPLOAD_FOLDER'] = 'static/profilepictures' 


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload_profile_picture', methods=['POST'])
def upload_profile_picture():
    username = session.get('username')  
    if not username:
        return jsonify({'error': 'You must be logged in to upload images'}), 403

    if 'profile_picture' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['profile_picture']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
        file.save(save_path)

        cursor = mysql.connection.cursor()
        cursor.execute('UPDATE Users SET ProfilePicture = %s WHERE Username = %s', (filename, username))
        mysql.connection.commit()
        
        flash('Profile picture uploaded successfully')
        return redirect(url_for('account'))
    else:
        flash('Invalid file type')
        return redirect(request.url)





@app.route('/klm_info')
def klm_info():
    return render_template('klm_info.html')







@app.route('/edit_details', methods=['GET', 'POST'])
def edit_details():
    if 'username' not in session:
        flash('You are not logged in.', 'danger')
        return redirect(url_for('login'))
    
    username = session['username']
    
    if request.method == 'POST':
        email = request.form['email']
        first_name = request.form['first_name']
        surname = request.form['surname']
        
        profile_picture_filename = None
        if 'profile_picture' in request.files:
            profile_picture = request.files['profile_picture']
            if profile_picture and allowed_file(profile_picture.filename):
                filename = secure_filename(profile_picture.filename)
                profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                profile_picture_filename = filename
        
        query = """
        UPDATE Users 
        SET Email = %s, FirstName = %s, Surname = %s
        WHERE Username = %s
        """
        params = (email, first_name, surname, username)

        if profile_picture_filename:
            query = """
            UPDATE Users 
            SET Email = %s, FirstName = %s, Surname = %s, ProfilePicture = %s
            WHERE Username = %s
            """
            params = (email, first_name, surname, profile_picture_filename, username)
        
        with mysql.connection.cursor() as cursor:
            cursor.execute(query, params)
            mysql.connection.commit()
        
        flash('Your details have been updated.', 'success')
        return redirect(url_for('account'))
    
    query = "SELECT Email, FirstName, Surname, ProfilePicture FROM Users WHERE Username = %s"
    
    with mysql.connection.cursor() as cursor:
        cursor.execute(query, (username,))
        user_details = cursor.fetchone()
    
    return render_template('edit_details.html', user_details=user_details)





if __name__ == '__main__':
    app.run(debug=True)



    
    