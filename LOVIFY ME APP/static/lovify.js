// Behaviour for the share controls.
//
// This file exists because the content security policy in security.py does
// not allow 'unsafe-inline' in script-src, so an onclick attribute would be
// silently dropped by the browser. That is the policy working: the same rule
// is what stops an injected <script> from running.

(function () {
  "use strict";

  // Tap a link field, get the whole link selected, ready to copy.
  document.querySelectorAll(".copy-field input").forEach(function (input) {
    input.addEventListener("focus", function () { input.select(); });
    input.addEventListener("click", function () { input.select(); });
  });

  // A real copy button where the browser supports it. No fallback needed:
  // the field above is selectable either way, so this is a convenience and
  // never the only route.
  if (!navigator.clipboard) return;

  document.querySelectorAll(".copy-field").forEach(function (field) {
    var input = field.querySelector("input");
    if (!input) return;

    var button = document.createElement("button");
    button.type = "button";
    button.className = "button copy-button";
    button.textContent = "Copy";
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(input.value).then(function () {
        button.textContent = "Copied";
        setTimeout(function () { button.textContent = "Copy"; }, 1800);
      });
    });
    field.appendChild(button);
  });
})();
