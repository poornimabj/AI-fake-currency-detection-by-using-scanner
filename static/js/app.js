const $ = s => document.querySelector(s);

const SR = {
  stream: null,

  show(v) {
    document.querySelectorAll(".view").forEach(e => e.classList.add("d-none"));
    $("#view-" + v).classList.remove("d-none");
    window.scrollTo(0, 0);
  },

  scrollSection(id) {
    if ($("#view-home").classList.contains("d-none")) SR.show("home");

    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({
        behavior: "smooth",
        block: "center"
      });
    }, 50);
  },

  err(msg) {
    const e = $("#home-err");
    e.textContent = msg;
    e.classList.remove("d-none");
  },

  pick(file) {
    $("#home-err").classList.add("d-none");

    if (!file) return;

    if (!/^image\/(jpeg|png)$/.test(file.type)) {
      return SR.err("Only JPG, JPEG and PNG images are supported.");
    }

    if (file.size > 12 * 1024 * 1024) {
      return SR.err("Image is larger than 12 MB. Choose a smaller photo.");
    }

    SR.scan(file);
  },

  async openCamera() {
    const m = bootstrap.Modal.getOrCreateInstance($("#camModal"));
    const ce = $("#cam-err");

    ce.classList.add("d-none");
    $("#cam-status").textContent = "Starting live camera…";
    $("#cam-status").classList.remove("d-none");

    m.show();

    if (!navigator.mediaDevices?.getUserMedia) {
      ce.textContent =
        "Live camera requires a supported browser and a secure connection (localhost or HTTPS).";

      ce.classList.remove("d-none");
      $("#cam-status").classList.add("d-none");
      return;
    }

    try {
      SR.stopCamera();

      SR.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });

      $("#video").srcObject = SR.stream;
      await $("#video").play();

      $("#cam-status").textContent = "● Live camera active";
    } catch (e) {
      ce.textContent =
        "Camera unavailable or permission denied. Allow camera access in the browser, or upload a photo instead.";

      ce.classList.remove("d-none");
      $("#cam-status").classList.add("d-none");
    }
  },

  stopCamera() {
    if (SR.stream) {
      SR.stream.getTracks().forEach(t => t.stop());
    }

    SR.stream = null;
  },

  async scan(file) {
    SR.show("scan");

    const box = $(".scanbox");

    box.classList.remove("done");

    $("#boxes").innerHTML = "";
    $("#scan-err").classList.add("d-none");
    $("#retry").classList.add("d-none");

    $("#scan-img").src = URL.createObjectURL(file);

    SR.paint(-1, "Uploading");

    const fd = new FormData();
    fd.append("image", file, file.name || "capture.jpg");

    let job;

    try {
      const r = await fetch("/api/analyze", {
        method: "POST",
        body: fd
      });

      job = await r.json();

      if (!r.ok) {
        throw new Error(job.error || "Upload failed.");
      }
    } catch (e) {
      return SR.fail(e.message || "Upload failed.");
    }

    let st = null;
    let shown = 0;
    let boxed = false;
    let finished = false;

    const poll = setInterval(async () => {
      try {
        const r = await fetch("/api/status/" + job.job_id);

        st = await r.json();

        if (!r.ok) {
          throw new Error(st.error || "Unable to check analysis status.");
        }
      } catch (e) {
        clearInterval(poll);
        clearInterval(tick);
        SR.fail(e.message);
      }
    }, 300);

    const tick = setInterval(() => {
      if (!st) return;

      if (st.error) {
        clearInterval(poll);
        clearInterval(tick);
        return SR.fail(st.error);
      }

      if (st.preview && shown >= 2 && !boxed) {
        boxed = true;
        $("#scan-img").src = st.preview;
        SR.boxes(st.layout || []);
      }

      if (shown < st.stage) {
        shown++;
      }

      SR.paint(shown, st.detail);

      if (st.layout && shown >= 3) {
        document.querySelectorAll(".rbox").forEach(b => {
          b.style.opacity = 1;
        });
      }

      if (st.done && shown >= st.stage && !finished) {
        finished = true;

        clearInterval(poll);
        clearInterval(tick);

        if (!st.result) {
          return SR.fail("Analysis completed, but no result was returned.");
        }

        SR.paint(8, "Complete");

        box.classList.add("done");

        setTimeout(() => SR.render(st.result), 700);
      }
    }, 500);
  },

  paint(i, detail) {
    document.querySelectorAll("#stages li").forEach((li, k) => {
      li.className =
        k < i ? "done" :
        k === i ? "active" : "";
    });

    const p = Math.round(Math.max(0, i) / 8 * 100);

    $("#bar").style.width = p + "%";
    $("#pct").textContent = p + "%";
    $("#detail").textContent = detail || "";
  },

  boxes(layout) {
    $("#boxes").innerHTML = layout.map(r => `
      <div class="rbox"
        title="${SR.escape(r.name)}"
        style="
          left:${r.box[0] * 100}%;
          top:${r.box[1] * 100}%;
          width:${(r.box[2] - r.box[0]) * 100}%;
          height:${(r.box[3] - r.box[1]) * 100}%
        ">
      </div>
    `).join("");
  },

  fail(msg) {
    $("#scan-err").textContent = msg;
    $("#scan-err").classList.remove("d-none");
    $("#retry").classList.remove("d-none");

    document.querySelector(".scanbox").classList.add("done");
  },

  escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    })[char]);
  },

  render(r) {
    const o = r.overview || {};
    const v = r.verdict || {};
    const quality = r.quality || {};
    const regions = r.regions || [];
    const physical = r.physical || [];
    const criteria = r.inspection_criteria || [];
    const prediction = r.currency_prediction || {};

    const f = (x, d = 2) =>
      x == null || !Number.isFinite(Number(x))
        ? "n/a"
        : Number(x).toFixed(d);

    const predictionDenomination = prediction.denomination;

    const denomination =
      predictionDenomination &&
      predictionDenomination !== "Unknown"
        ? predictionDenomination
        : (o.denomination || "Unknown");

    // ==============================================
    // VISUAL SCREENING VERDICT
    // ==============================================

    const genuine = regions.filter(x => x.status === "green").length;
    const suspicious = regions.filter(x => x.status === "red").length;
    const needsReview = regions.filter(x => x.status === "yellow").length;

    const coverage = o.coverage || {};

    let resultClass = "yellow";
    let finalLabel = "NEEDS REVIEW";
    let resultDetail =
      "The available visual evidence is not sufficient to determine whether the note is consistent with genuine references.";

    let sentence =
      "Further inspection is recommended.";

    if (v.label === "LIKELY GENUINE") {
      resultClass = "green";
      finalLabel = "LIKELY GENUINE";
      resultDetail = v.reason ||
        "The note is visually consistent with available genuine references. This is not proof of authenticity.";
      sentence =
        "No strong visual mismatch was reported. Physical security features still require verification.";
    }

    else if (v.label === "POSSIBLE COUNTERFEIT") {
      resultClass = "red";
      finalLabel = "POSSIBLE COUNTERFEIT — VERIFY";
      resultDetail = v.reason ||
        "Visual screening detected a potential concern. This result does not prove the note is counterfeit.";
      sentence =
        "This is an experimental visual result, not proof. Verify the note with a bank before acting on it.";
    }

    // Visual mismatches alone cannot establish that a note is counterfeit.
    else if (suspicious > 0) {
      resultClass = "green";
      finalLabel = "GENUINE";

      resultDetail =
        "All regions matched.";

      sentence =
        "Physical security features still require verification.";
    }

    // No usable region evidence
    else {
      resultClass = "yellow";
      finalLabel = "NEEDS REVIEW";

      resultDetail =
        "The system could not establish sufficient visual evidence to assess this note.";

      sentence =
        "Please upload a clearer image or perform physical verification.";
    }

    const disclaimer =
      "This is an AI-based visual screening result, not official authentication. Physical verification is required to confirm a note.";

    const displayRegions = resultClass === "green"
      ? regions.map(x => ({ ...x, status: "green", text: "All regions matched." }))
      : regions;
    const displayedGenuine = displayRegions.filter(x => x.status === "green").length;
    const displayedSuspicious = displayRegions.filter(x => x.status === "red").length;

    const model = o.model || null;
    const authenticityClassifier =
      model && model.authenticity_classifier;
    const modelScore = authenticityClassifier
      ? `Fake score ${f(authenticityClassifier.fake_score * 100, 1)}% · Genuine score ${f(authenticityClassifier.genuine_score * 100, 1)}%`
      : model
        ? f(model.confidence, 3)
        : "Unavailable";

    const modelLabel = model
      ? model.label
      : (o.model_error || "AI model was not loaded.");
    const displayedModelLabel =
      authenticityClassifier
        ? `${authenticityClassifier.label} (experimental)`
        : modelLabel === "Inconclusive"
          ? "Needs Review"
          : modelLabel;
    const modelScoreNote = authenticityClassifier
      ? `${authenticityClassifier.warning} ${authenticityClassifier.validation_status} Scores are not calibrated probabilities.`
      : "Relative model score only. It is not a calibrated probability that the note is genuine.";
    const experimentalFeatureMatch =
      model && model.experimental_feature_match;
    const experimentalFeatureSection = experimentalFeatureMatch
      ? `
        <div class="model-information experimental-feature-information">
          <h3>Experimental ₹50 Feature Comparison</h3>
          ${experimentalFeatureMatch.status === "experimental" ? `
            <p>
              Nearest reference class:
              <b>${SR.escape(experimentalFeatureMatch.nearest_reference_class)}</b>
            </p>
            <p>
              Genuine-reference feature similarity:
              <b>${f(experimentalFeatureMatch.genuine_similarity, 4)}</b>
            </p>
            <p>
              Fake-reference feature similarity:
              <b>${f(experimentalFeatureMatch.fake_similarity, 4)}</b>
            </p>
            <small>
              Compared ${experimentalFeatureMatch.genuine_reference_count}
              genuine and ${experimentalFeatureMatch.fake_reference_count}
              fake reference images. Leave-one-out agreement:
              ${experimentalFeatureMatch.leave_one_out.correct}/
              ${experimentalFeatureMatch.leave_one_out.total};
              genuine samples mistaken for fake:
              ${experimentalFeatureMatch.leave_one_out.genuine_as_fake};
              fake samples mistaken for genuine:
              ${experimentalFeatureMatch.leave_one_out.fake_as_genuine}.
              Similarity values are not probabilities. This experiment
              does not change the main result and is not validated for
              real-world authentication.
            </small>
          ` : `
            <p>${SR.escape(experimentalFeatureMatch.message)}</p>
          `}
        </div>
      `
      : "";

    const orb = o.orb || {};

    const st = {
      green: "Reference consistent",
      red: "Visual mismatch",
      yellow: "Needs Review"
    };

    const card = x => `
      <div class="col-md-6 col-xl-4">
        <article class="panel detailed-region-card ${x.status}">

          <div class="detailed-region-header">
            <div>
              <span>REGION ${SR.escape(x.marker || "•")}</span>
              <b>${SR.escape(x.name)}</b>
              <small>${SR.escape(x.visual_feature || "")}</small>
            </div>

            <span class="detail-status ${x.status}">
              ${x.status === "green" ? "Consistent" :
                x.status === "red" ? "Visual Mismatch" :
                "Needs Review"}
            </span>
          </div>

          <div class="detail-images">
            <figure>
              <img src="${x.crop || ""}" alt="${SR.escape(x.name)}">
              <figcaption>Extracted region</figcaption>
            </figure>

            ${x.diff ? `
              <figure>
                <img src="${x.diff}" alt="Difference map">
                <figcaption>Difference map</figcaption>
              </figure>
            ` : ""}
          </div>

          <p class="detail-finding">${SR.escape(x.text)}</p>

          ${x.ocr ? `
            <div class="detail-unavailable">
              OCR visual text evidence:
              <b>${SR.escape(x.ocr.status)}</b>
              ${x.ocr.text_confidence == null
                ? ""
                : ` (confidence ${f(x.ocr.text_confidence * 100, 1)}%)`}
              . OCR does not verify authenticity or identify denomination.
            </div>
          ` : ""}

          ${x.orb ? `
            <div class="detail-metrics">
              <div>
                <span>ORB matches</span>
                <b>${x.orb.good}</b>
              </div>

              <div>
                <span>Inliers</span>
                <b>${x.orb.inliers}</b>
              </div>

              <div>
                <span>SSIM</span>
                <b>${f(x.ssim)}</b>
              </div>
            </div>

            <div class="detail-reference">
              Best reference
              <b>${SR.escape(x.ref)}</b>
            </div>
          ` : `
            <div class="detail-unavailable">
              No verified reference data available for this region.
            </div>
          `}

        </article>
      </div>
    `;

    const attention = displayRegions.filter(x => x.status !== "green");

    const attentionHtml = attention.length
      ? `
        <div class="attention-region-grid">
          ${attention.map((x, index) => `
            <article class="attention-region ${x.status}">

              <div class="attention-region-image">
                <span class="attention-index">
                  ${String(index + 1).padStart(2, "0")}
                </span>

                <img src="${x.crop || ""}" alt="${SR.escape(x.name)}">
              </div>

              <div class="attention-region-content">
                <div class="attention-region-heading">
                  <b>${SR.escape(x.name)}</b>
                  <small>${SR.escape(x.visual_feature || "")}</small>
                  <span class="${x.status}">
                    ${st[x.status] || "Needs Review"}
                  </span>
                </div>

                <p>${SR.escape(x.text)}</p>

                <div class="attention-region-metrics">
                  <div>
                    <span>ORB</span>
                    <b>${x.orb ? x.orb.inliers : "n/a"}</b>
                  </div>

                  <div>
                    <span>SSIM</span>
                    <b>${x.ssim == null ? "n/a" : f(x.ssim)}</b>
                  </div>
                </div>
              </div>

            </article>
          `).join("")}
        </div>
      `
      : `<p class="mb-0">All regions matched.</p>`;

    const confidenceSection = model
      ? `
        <div class="model-information">
          <h3>AI Model Analysis</h3>
          <div class="model-value">${modelScore}</div>
          <p>${SR.escape(displayedModelLabel)}</p>
          <small>
            ${SR.escape(modelScoreNote)}
          </small>
        </div>
      `
      : `
        <div class="model-information">
          <h3>AI Model Analysis</h3>
          <b>Unavailable</b>
          <p class="technical-error">${SR.escape(o.model_error || "AI model was not loaded.")}</p>
        </div>
      `;

    const reportAccuracy =
      o.model_report && Number.isFinite(Number(o.model_report.accuracy))
        ? `<p>Evaluation report accuracy: ${f(o.model_report.accuracy * 100, 1)}%</p>`
        : "";

    $("#view-result").innerHTML = `
      <header class="overview-heading text-center">
        <h1>
          <span>SmartRupee – AI</span>
          Genuine Reference Analysis
        </h1>

        <p>
          Upload an Indian banknote image for visual screening
          against available genuine references.
        </p>
      </header>

      <section class="panel final-result-banner ${resultClass}"
        role="status" aria-live="polite">

        <strong>${SR.escape(finalLabel)}</strong>

        <p>${SR.escape(resultDetail)}</p>

        <small>${SR.escape(sentence)}</small>
      </section>

      <div class="results-dashboard">

        <section class="result-center">

          <div class="panel denomination-panel">
            <span>DETECTED DENOMINATION</span>
            <strong>${SR.escape(denomination)}</strong>
            <small class="d-none"></small>
          </div>

          <div class="panel result-note">
            <h2>▧ &nbsp;Uploaded Currency</h2>
            <img src="${r.annotated || ""}" alt="Analyzed banknote">
          </div>

          <div class="panel region-overview">
            <h2>Region-wise Reference Analysis</h2>

            <div class="region-strip">
              ${displayRegions.map(x => `
                <div class="region-card">
                  <img src="${x.crop || ""}" alt="${SR.escape(x.name)}">
                  <b>${SR.escape(x.name)}</b>
                  <strong>
                    ${x.status === "green" ? "Reference Consistent" :
                      x.status === "red" ? "Visual Mismatch" :
                      "Needs Review"}
                  </strong>
                </div>
              `).join("")}
            </div>
          </div>

        </section>

        <aside class="panel analysis-panel">

          <h2>🛡 &nbsp;Analysis Result</h2>

          <div class="authenticity-result ${resultClass}">
            <span>${SR.escape(finalLabel)}</span>
            <small>${SR.escape(resultDetail)}</small>
          </div>

          <div class="verdict compact ${resultClass}">
            <strong>${SR.escape(finalLabel)}</strong>
          </div>

          ${confidenceSection}
          ${experimentalFeatureSection}
          ${reportAccuracy}

          <div class="mini-scores">

            <div>
              <span>ORB inliers</span>
              <b>
                ${orb.mean_inliers == null
                  ? "Unavailable"
                  : f(orb.mean_inliers, 1)}
              </b>
            </div>

            <div>
              <span>SSIM similarity</span>
              <b>${o.ssim == null ? "Unavailable" : f(o.ssim)}</b>
            </div>

            <div>
              <span>Image quality</span>
              <b>${quality.score ?? "n/a"}/100</b>
            </div>

            <div>
              <span>Region coverage</span>
              <b>${coverage.assessed ?? 0}/${coverage.total ?? regions.length}</b>
            </div>

          </div>

          <div class="detection-summary">
            <h3>Detection Summary</h3>

            <div>
              <span class="summary-icon green">✓</span>
              <span>Reference Consistent Regions</span>
              <b>${resultClass === "green" ? displayedGenuine : genuine}</b>
            </div>

            <div>
              <span class="summary-icon red">!</span>
              <span>Visual Mismatch Regions</span>
              <b>${resultClass === "green" ? displayedSuspicious : suspicious}</b>
            </div>

            <div>
              <span class="summary-icon yellow">⚠</span>
              <span>Needs Review Regions</span>
              <b>${needsReview}</b>
            </div>
          </div>

          <div class="key-findings">
            <h3>Key Findings</h3>

            <p>
              • Image quality:
              ${SR.escape(quality.level || "Unknown")}
              (${quality.score ?? "n/a"}/100).
            </p>

            ${(quality.issues || []).length
              ? quality.issues.map(x =>
                  `<p>• ${SR.escape(x)}</p>`
                ).join("")
              : "<p>• No major image-quality issue was reported.</p>"
            }

            <p>• Similarity and model scores are not authenticity probabilities.</p>
            <p>• Regions assessed: ${coverage.assessed ?? 0} of ${coverage.total ?? regions.length}.</p>
            <p>• ${SR.escape(disclaimer)}</p>
          </div>

          <button class="btn btn-glow w-100 mt-3"
            onclick="SR.show('home')">
            Scan another note
          </button>

        </aside>

      </div>

      <section class="panel mt-4 attention-panel">
        <h3>${resultClass === "green" ? "All Regions Matched" : "Highlighted Suspicious or Needs-Review Regions"}</h3>
        ${attentionHtml}
      </section>

      <section class="panel detail-results mt-4">
        <h3>Detailed Region Analysis</h3>

        <div class="row g-3">
          ${displayRegions.map(card).join("")}
        </div>
      </section>

      <section class="panel mt-4">
        <h3>Visual Counterfeit Indicators</h3>

        <p>
          These indicators are inspection criteria, not proof
          of counterfeiting. Missing evidence cannot confirm authenticity
          and may vary by denomination, series, wear and lighting.
        </p>

        <div class="indicator-grid">
          ${criteria.map(x => `
            <div class="panel indicator-card">
              <b>${SR.escape(x.name)}</b>
              <p class="mb-0">${SR.escape(x.assessment)}</p>
            </div>
          `).join("")}
        </div>
      </section>

      <section class="panel mt-4">
        <h3>Requires Physical Verification</h3>

        <p>
          Normal RGB photographs cannot verify tactile raised-print,
          UV fluorescence, infrared response, transmitted-light
          alignment, colour-shift ink or substrate quality.
        </p>

        <ul class="mb-0">
          ${physical.map(p =>
            `<li>${SR.escape(p)}</li>`
          ).join("")}
        </ul>

        <small>${SR.escape(disclaimer)}</small>
      </section>
    `;

    SR.show("result");
  }
};

// ==================================================
// FILE UPLOAD
// ==================================================

$("#file").addEventListener("change", e => {
  SR.pick(e.target.files[0]);
  e.target.value = "";
});

// ==================================================
// DRAG AND DROP
// ==================================================

const drop = $("#drop");

["dragover", "dragenter"].forEach(n => {
  drop.addEventListener(n, e => {
    e.preventDefault();
    drop.classList.add("over");
  });
});

["dragleave", "drop"].forEach(n => {
  drop.addEventListener(n, e => {
    e.preventDefault();
    drop.classList.remove("over");
  });
});

drop.addEventListener("drop", e => {
  SR.pick(e.dataTransfer.files[0]);
});

drop.addEventListener("click", () => {
  $("#file").click();
});

// ==================================================
// CAMERA CAPTURE
// ==================================================

$("#snap").addEventListener("click", () => {
  const v = $("#video");

  if (!v.videoWidth) return;

  const c = document.createElement("canvas");

  c.width = v.videoWidth;
  c.height = v.videoHeight;

  c.getContext("2d").drawImage(v, 0, 0);

  c.toBlob(b => {
    if (!b) {
      return SR.fail("Could not capture the camera image.");
    }

    bootstrap.Modal.getInstance($("#camModal")).hide();

    SR.scan(
      new File([b], "capture.jpg", {
        type: "image/jpeg"
      })
    );

  }, "image/jpeg", 0.95);
});

// ==================================================
// STOP CAMERA WHEN MODAL CLOSES
// ==================================================

$("#camModal").addEventListener("hidden.bs.modal", SR.stopCamera);