/* Workspace interactions: file choice, preview, submit, state announcements.
 * No framework. Progressive: the form still posts without JS, and every state change is announced
 * in the live region so a screen reader hears it. */
(function () {
  "use strict";

  var form = document.getElementById("analyze-form");
  var input = document.getElementById("image-input");
  var dropzone = document.getElementById("dropzone");
  var analyzeBtn = document.getElementById("analyze-button");
  var resetBtn = document.getElementById("reset-button");
  var fileHint = document.getElementById("file-hint");
  var previewWrap = document.getElementById("preview-wrap");
  var previewImage = document.getElementById("preview-image");
  var previewMeta = document.getElementById("preview-meta");
  var live = document.getElementById("live-region");

  var panes = {
    idle: document.getElementById("features-idle"),
    loading: document.getElementById("features-loading"),
    error: document.getElementById("features-error"),
    result: document.getElementById("features-result")
  };
  var errorMessage = document.getElementById("features-error-message");
  var featuresChip = document.getElementById("features-chip");
  var featuresTable = document.getElementById("features-table");
  var topTable = document.getElementById("top-table");
  var provenance = document.getElementById("features-provenance");

  if (!form || !input) { return; }

  function say(message) { if (live) { live.textContent = message; } }

  function show(which) {
    Object.keys(panes).forEach(function (key) {
      if (panes[key]) { panes[key].hidden = key !== which; }
    });
    if (featuresChip) { featuresChip.hidden = which !== "result"; }
  }

  function kb(bytes) { return (bytes / 1024).toFixed(0) + " KB"; }

  function setFile(file) {
    if (!file) { return; }
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
    fileHint.textContent = file.name + ", " + kb(file.size);
    var url = URL.createObjectURL(file);
    previewImage.src = url;
    /* The uploaded image is the person's own, so a generic alt would be a lie about its content.
       It is labelled as the uploaded file, and the generated description is page content. */
    previewImage.alt = "The image you uploaded: " + file.name;
    previewImage.onload = function () {
      previewMeta.textContent = previewImage.naturalWidth + " by " +
        previewImage.naturalHeight + " pixels, " + kb(file.size);
      URL.revokeObjectURL(url);
    };
    previewWrap.hidden = false;
    show("idle");
    say("Image chosen: " + file.name + ". Ready to extract features.");
  }

  function clearFile() {
    input.value = "";
    analyzeBtn.disabled = true;
    resetBtn.hidden = true;
    fileHint.textContent = "No file chosen yet.";
    previewWrap.hidden = true;
    previewImage.removeAttribute("src");
    previewImage.alt = "";
    show("idle");
    say("Image cleared.");
    input.focus();
  }

  input.addEventListener("change", function () { setFile(input.files && input.files[0]); });
  resetBtn.addEventListener("click", clearFile);

  /* The dropzone is a <label for> the input, so click and keyboard already work through the label.
     These handlers only add pointer drag and drop on top of that. */
  ["dragenter", "dragover"].forEach(function (evt) {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropzone.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    dropzone.addEventListener(evt, function () { dropzone.classList.remove("is-over"); });
  });
  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) {
      input.files = files;
      setFile(files[0]);
    }
  });

  function row(label, value) {
    var tr = document.createElement("tr");
    var th = document.createElement("th");
    th.setAttribute("scope", "row");
    th.textContent = label;
    var td = document.createElement("td");
    td.textContent = value;
    tr.appendChild(th);
    tr.appendChild(td);
    return tr;
  }

  function render(data) {
    var f = data.features, img = data.image;
    featuresTable.replaceChildren(
      row("Image", img.format + ", " + img.width + " by " + img.height + " pixels, " + kb(img.bytes)),
      row("Full vector", f.full.dim + " dimensions"),
      row("Full vector: mean / max", f.full.mean + " / " + f.full.max),
      row("Full vector: zero fraction", f.full.sparsity),
      row("Full vector: L2 norm", f.full.l2_norm),
      row("After Fisher selection", f.reduced.dim + " dimensions"),
      row("Selected: mean / max", f.reduced.mean + " / " + f.reduced.max),
      row("Selected: zero fraction", f.reduced.sparsity),
      row("Selected: L2 norm", f.reduced.l2_norm),
      row("Inference time", f.inference_seconds + " s"),
      row("Model load time", (f.model_load_seconds === null ? "already loaded"
        : f.model_load_seconds + " s (first run in this process)"))
    );
    topTable.replaceChildren.apply(topTable, f.top_activations.map(function (t) {
      var tr = document.createElement("tr");
      [t.selected_position, t.source_index, t.activation].forEach(function (v) {
        var td = document.createElement("td");
        td.textContent = v;
        tr.appendChild(td);
      });
      return tr;
    }));
    provenance.textContent = "Preprocessing: " + f.preprocessing +
      ". Code path: " + f.source + ". Selection read from " + f.selection_source +
      ", which is not refitted here.";
    show("result");
    say("Feature extraction finished. " + f.full.dim + " dimensions reduced to " +
      f.reduced.dim + " in " + f.inference_seconds + " seconds. No caption was generated, " +
      "because caption generation is not implemented.");
  }

  function fail(message) {
    errorMessage.textContent = message;
    show("error");
    say("Feature extraction failed. " + message);
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!input.files || !input.files.length) { return; }
    show("loading");
    analyzeBtn.disabled = true;
    say("Extracting features. This can take a few seconds on the first run.");

    var body = new FormData();
    body.append("image", input.files[0]);

    fetch(form.action, { method: "POST", body: body })
      .then(function (response) {
        return response.json().then(function (data) { return { ok: response.ok, data: data }; });
      })
      .then(function (r) {
        if (r.ok && r.data.ok) { render(r.data); }
        else { fail(r.data && r.data.error ? r.data.error : "The server returned an error."); }
      })
      .catch(function () {
        fail("The request could not be completed. Check that the server is still running.");
      })
      .then(function () { analyzeBtn.disabled = false; });
  });
})();
