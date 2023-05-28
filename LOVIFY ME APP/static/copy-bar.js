var copyButton = document.querySelector('.copy-button');
var copiedMessage = document.querySelector('.copied-message');
var inputField = document.querySelector('.url');

copyButton.addEventListener('click', function() {
    inputField.select();
    document.execCommand("copy");
    copiedMessage.style.display = 'block';
    setTimeout(function() {
        copiedMessage.style.display = 'none';
    }, 1000);
});