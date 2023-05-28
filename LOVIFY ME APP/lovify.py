import uuid
import os
import requests
from base64 import urlsafe_b64encode
import json
from urllib.parse import urlencode

from dotenv import load_dotenv

from flask import Flask, request, session, redirect, render_template, url_for, make_response, send_from_directory
from flask_session import Session

load_dotenv()

auth_url = "https://accounts.spotify.com/authorize"
token_url = "https://accounts.spotify.com/api/token"

app = Flask(__name__, template_folder="templates", static_url_path="/static")
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)

app.secret_key = "K4%jge$i9!ov6CrW8^y$28%$@ktzNLFTy"
app.config['SESSION_COOKIE_NAME'] = "Lovify My Cookies"
app.config['DEFAULT_USER_IMAGE'] = 'default_user_image.svg'

Session(app)

client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
scope = os.getenv('SCOPE')

redirect_uri_default = "http://localhost:5000/redirect-user"

cache_folder = './cache/'
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

def session_cache_path():
    uuid = session.get('uuid')
    return cache_folder + (uuid if uuid is not None else '')

def authorize_user(redirect_uri=None):
    if not session.get('uuid'):
        session['uuid'] = str(uuid.uuid4())

    state = session['uuid']
    payload = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri or redirect_uri_default,
        "scope": scope,
        "state": state,
    }

    auth_url_params = urlencode(payload)
    auth_request_url = f"{auth_url}?{auth_url_params}"
    return auth_request_url

def get_access_token(code, redirect_uri=None):
    if redirect_uri is None:
        redirect_uri = redirect_uri_default
        
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(token_url, data=payload, headers=headers)
    token_data = response.json()

    return token_data.get("access_token")

def revoke_user_token(token):
    if not token:
        return
    headers = {"Authorization": f"Bearer {token}"}
    requests.post("https://accounts.spotify.com/api/token/revoke", headers=headers, data={"token": token})

@app.route('/login')
def login_redirect():
    auth = authorize_user()
    return redirect(auth)

@app.route('/second/user/login')
def login_two_redirect():
    custom_redirect_uri = "http://localhost:5000/redirect-second-user"
    auth = authorize_user(redirect_uri=custom_redirect_uri)
    return redirect(auth)

@app.route('/')
def home_page():
    token = session.get('token')
    revoke_user_token(token)
    session.clear()
    return render_template('index.html')

@app.route('/what-about-us')
def about_page():
    return render_template('about.html')

@app.route('/find-compatibility')
def compatibility_page():
    first_user_info = session.get('first_user_info', {})
    second_user_info = session.get('second_user_info', {})
    return render_template('compatibilityV2.html', first_user_info=first_user_info, second_user_info=second_user_info)

@app.route('/redirect-user')
def logged_in_user():
    code = request.args.get("code")
    if code:
        access_token = get_access_token(code, redirect_uri=redirect_uri_default)
        session["token"] = access_token

    if "token" not in session:
        return redirect(url_for('login_redirect'))

    headers = {"Authorization": f"Bearer {session['token']}"}
    user_info_url = "https://api.spotify.com/v1/me"
    user_info_response = requests.get(user_info_url, headers=headers)
    user = user_info_response.json()

    try:
        user_name = user['display_name']
        user_uri = user['uri']
        if user['images']:
            user_photo = user['images'][0]['url']
        else:
            user_photo = url_for('static', filename='default_user_image.svg')
    except KeyError:
        return redirect(url_for('login_redirect'))

    session['first_user_info'] = {'user_name': user_name, 'user_photo': user_photo, 'user_uri': user_uri}

    response = make_response(render_template("user_index.html", user_name=user_name, user_photo=user_photo, user_uri=user_uri))
    response.set_cookie("loggedIn", "true")
    return response

@app.route('/redirect-second-user')
def logged_in_second_user():
    code = request.args.get("code")
    if code:
        access_token = get_access_token(code, redirect_uri="http://localhost:5000/redirect-second-user")
        session["token"] = access_token

    if "token" not in session:
        return redirect(url_for('login_two_redirect'))

    headers = {"Authorization": f"Bearer {session['token']}"}
    user_info_url = "https://api.spotify.com/v1/me"
    user_info_response = requests.get(user_info_url, headers=headers)
    user = user_info_response.json()

    try:
        user_name = user['display_name']
        user_uri = user['uri']
        if user['images']:
            user_photo = user['images'][0]['url']
        else:
            user_photo = url_for('static', filename='default_user_image.svg')
    except KeyError:
        return redirect(url_for('login_two_redirect'))

    session['second_user_info'] = {'user_name': user_name, 'user_photo': user_photo, 'user_uri': user_uri}

    return redirect(url_for('compatibility_page'))

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port='5000')