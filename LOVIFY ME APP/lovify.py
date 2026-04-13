import uuid
import os
import requests
from urllib.parse import urlencode

from dotenv import load_dotenv

from flask import Flask, request, session, redirect, render_template, url_for, make_response

load_dotenv()

auth_url = "https://accounts.spotify.com/authorize"
token_url = "https://accounts.spotify.com/api/token"

app = Flask(__name__, template_folder="templates", static_url_path="/static")

app.secret_key = os.getenv('SECRET_KEY', "K4%jge$i9!ov6CrW8^y$28%$@ktzNLFTy")
app.config['SESSION_COOKIE_NAME'] = "lovify_session"

client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
scope = os.getenv('SCOPE')

# BASE_URL: set in .env for production (e.g. https://yourapp.onrender.com)
# Falls back to local dev server
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:8080')
REDIRECT_URI_USER1 = f"{BASE_URL}/redirect-user"
REDIRECT_URI_USER2 = f"{BASE_URL}/redirect-second-user"


def build_auth_url(redirect_uri):
    """Build Spotify authorization URL and store state in session."""
    if not session.get('uuid'):
        session['uuid'] = str(uuid.uuid4())

    payload = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": session['uuid'],
    }
    return f"{auth_url}?{urlencode(payload)}"


def exchange_code_for_token(code, redirect_uri):
    """Exchange authorization code for access + refresh tokens."""
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    response = requests.post(token_url, data=payload,
                             headers={"Content-Type": "application/x-www-form-urlencoded"})

    if not response.ok:
        app.logger.warning(f"Token exchange failed: {response.status_code} {response.text}")
        return None, None

    data = response.json()
    if "error" in data:
        app.logger.warning(f"Token exchange error: {data}")
        return None, None

    return data.get("access_token"), data.get("refresh_token")


def fetch_user_profile(access_token):
    """Fetch the current user's Spotify profile."""
    response = requests.get("https://api.spotify.com/v1/me",
                            headers={"Authorization": f"Bearer {access_token}"})

    if response.status_code != 200:
        app.logger.warning(f"/me failed: {response.status_code} {response.text[:200]}")
        return None

    return response.json()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/login')
def login_redirect():
    # Fresh state for every login attempt
    session['uuid'] = str(uuid.uuid4())
    return redirect(build_auth_url(REDIRECT_URI_USER1))


@app.route('/second/user/login')
def login_two_redirect():
    # Keep first user's data but refresh the state for this auth flow
    session['uuid'] = str(uuid.uuid4())
    return redirect(build_auth_url(REDIRECT_URI_USER2))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home_page'))


@app.route('/what-about-us')
def about_page():
    return render_template('about.html')


@app.route('/dashboard')
def dashboard():
    """Logged-in user's dashboard. Reads from session, safe to refresh."""
    user_info = session.get('first_user_info')
    if not user_info:
        return redirect(url_for('login_redirect'))

    return render_template("user_index.html",
                           user_name=user_info['user_name'],
                           user_photo=user_info['user_photo'],
                           user_uri=user_info['user_uri'],
                           share_link=f"{BASE_URL}/second/user/login")


@app.route('/find-compatibility')
def compatibility_page():
    first_user_info = session.get('first_user_info', {})
    second_user_info = session.get('second_user_info', {})
    return render_template('compatibilityV2.html',
                           first_user_info=first_user_info,
                           second_user_info=second_user_info)


# ─── OAuth Callbacks ──────────────────────────────────────────────────────────

@app.route('/redirect-user')
def logged_in_user():
    # Validate state
    state = request.args.get("state")
    if state != session.get('uuid'):
        app.logger.warning(f"State mismatch: url={state} session={session.get('uuid')}")
        return redirect(url_for('home_page'))

    if request.args.get("error"):
        return redirect(url_for('home_page'))

    code = request.args.get("code")
    if not code:
        return redirect(url_for('home_page'))

    # Exchange code for token
    access_token, refresh_token = exchange_code_for_token(code, REDIRECT_URI_USER1)
    if not access_token:
        return redirect(url_for('home_page'))

    # Fetch profile
    user = fetch_user_profile(access_token)
    if not user:
        return redirect(url_for('home_page'))

    try:
        user_name = user['display_name']
        user_uri = user['uri']
        user_photo = (user['images'][0]['url'] if user.get('images')
                      else url_for('static', filename='default_user_image.svg'))
    except (KeyError, IndexError):
        return redirect(url_for('home_page'))

    # Save to session and redirect to dashboard (safe to refresh)
    session['first_user_token'] = access_token
    session['first_user_refresh'] = refresh_token
    session['first_user_info'] = {
        'user_name': user_name,
        'user_photo': user_photo,
        'user_uri': user_uri,
    }

    return redirect(url_for('dashboard'))


@app.route('/redirect-second-user')
def logged_in_second_user():
    state = request.args.get("state")
    if state != session.get('uuid'):
        return redirect(url_for('home_page'))

    if request.args.get("error"):
        return redirect(url_for('compatibility_page'))

    code = request.args.get("code")
    if not code:
        return redirect(url_for('home_page'))

    access_token, refresh_token = exchange_code_for_token(code, REDIRECT_URI_USER2)
    if not access_token:
        return redirect(url_for('home_page'))

    user = fetch_user_profile(access_token)
    if not user:
        return redirect(url_for('home_page'))

    try:
        user_name = user['display_name']
        user_uri = user['uri']
        user_photo = (user['images'][0]['url'] if user.get('images')
                      else url_for('static', filename='default_user_image.svg'))
    except (KeyError, IndexError):
        return redirect(url_for('home_page'))

    session['second_user_token'] = access_token
    session['second_user_refresh'] = refresh_token
    session['second_user_info'] = {
        'user_name': user_name,
        'user_photo': user_photo,
        'user_uri': user_uri,
    }

    return redirect(url_for('compatibility_page'))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='8080')
