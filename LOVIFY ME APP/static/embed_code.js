const trackIds = [
    "0VjIjW4GlUZAMYd2vXMi3b",
     "6v3KW9xbzN5yKLt9YKDYA2",
     "2Fxmhks0bxGSBdJ92vM42m",
     "3KkXRkHbMCARz0aVfEt68P",
     "3tjFYV6RSFtuktYl3ZtYcq", 
     "7xGfFoTpQ2E7fRF5lN10tr",
     "6PLeTn7dY8WnTf7v6ZzEGN", 
     "0Wt15NwMa6IyQ63l0d9XkD",
     "2azA5UJ3kVqk3qqlc7YP9X"
    ];

// Spotify embed code generation
function generateSpotifyEmbed(trackId) {
  return `
    <div class="spotify-track">
      <iframe src="https://open.spotify.com/embed/track/${trackId}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    </div>
  `;
}

// Populate the scrollable section with embedded tracks
function populateSpotifyTracks() {
  const container = document.getElementById("spotify-tracks");

  // Generate embedded tracks and append to the container
  trackIds.forEach((trackId) => {
    const embedCode = generateSpotifyEmbed(trackId);
    container.innerHTML += embedCode;
  });
}

// Initialize the Spotify SDK and populate the tracks
window.onSpotifyWebPlaybackSDKReady = () => {
  populateSpotifyTracks();
};