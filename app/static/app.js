// Progressive enhancement only: every action on the page also works as a
// plain HTML form post. This script adds client-side file validation and a
// lazily loaded JSON view of the current result.
(function () {
  "use strict";

  var MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

  document.querySelectorAll(".upload-grid").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      var ref = form.querySelector('input[name="ref_file"]');
      var hyp = form.querySelector('input[name="hyp_file"]');
      var files = [ref, hyp].filter(Boolean).map(function (input) {
        return input.files && input.files[0];
      });
      if (files.length !== 2 || files.some(function (f) { return !f; })) {
        return; // Let the server explain the problem.
      }
      var tooBig = files.find(function (f) { return f.size > MAX_UPLOAD_BYTES; });
      if (tooBig) {
        event.preventDefault();
        window.alert("文件 " + tooBig.name + " 超过 4 MB，请先精简后再上传。");
        return;
      }
      var button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.textContent = "评分中…";
      }
    });
  });

  var details = document.querySelector(".raw-panel");
  if (!details) return;
  var pre = document.getElementById("raw-json");
  var loaded = false;
  details.addEventListener("toggle", function () {
    if (!details.open || loaded || !pre) return;
    loaded = true;
    fetch("/api/result", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        pre.textContent = JSON.stringify(data, null, 2);
      })
      .catch(function (err) {
        pre.textContent = "读取结果失败：" + err.message;
      });
  });
})();
