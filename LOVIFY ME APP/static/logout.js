function logoutAndRedirect() {
    var logoutUrl = "https://accounts.spotify.com/en/logout";
    var redirectUrl = "http://localhost:5000/loivfy/champions";
    var win = window.open(logoutUrl, "Logout", "width=400,height=500");

    var pollTimer = window.setInterval(function () {
        try {
            // Check if the window is on the Spotify logout status page
            if (win.document.URL.indexOf("https://accounts.spotify.com/en/status") !== -1) {
                window.clearInterval(pollTimer);
                win.close();
                window.location.href = redirectUrl;
            }
        } catch (e) {
        }
    }, 100);
}