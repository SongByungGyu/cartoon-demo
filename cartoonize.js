/**
 * Cartoon Demo — TensorFlow.js 기반 카툰 변환
 *
 * 동작:
 *   - unpkg CDN 에 배포된 CartoonGAN TF.js 모델 4종 로드
 *   - 스타일 선택 시 해당 모델로 즉시 변환
 *   - 안드로이드 WebView 에서 오는 이미지를 CartoonBridge 로 받음
 *   - 결과를 CartoonBridge.onResult 로 반환
 *
 * 모델 소스: mnicnc404/CartoonGan-tensorflow (본래 TF SavedModel)
 * TF.js 변환본 CDN: https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/*
 */

// 모델 URL (unpkg CDN)
const MODEL_URLS = {
    hayao:   "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/hayao/model.json",
    shinkai: "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/shinkai/model.json",
    hosoda:  "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/hosoda/model.json",
    paprika: "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/paprika/model.json",
};

// 로드된 모델 캐시 (한 번 로드하면 재사용)
const modelCache = {};

// 현재 상태
let currentStyle = "hayao";
let originalImageData = null;  // base64 dataUrl

// DOM
const statusEl = document.getElementById("status");
const stylePickerEl = document.getElementById("stylePicker");
const originalImgEl = document.getElementById("original");
const cartoonImgEl = document.getElementById("cartoon");
const loadingEl = document.getElementById("loading");
const fileInputEl = document.getElementById("fileInput");

// ─────────────────────────────────────────────
// 초기화
// ─────────────────────────────────────────────
init();

async function init() {
    setStatus("TensorFlow.js 초기화 중...");
    await tf.ready();
    const backend = tf.getBackend();
    setStatus(`백엔드: ${backend}. 사용 준비 완료.`, "success");

    setupStylePicker();
    setupFileInput();

    // 안드로이드 WebView 안에서 실행 중인지 감지
    if (typeof CartoonBridge !== "undefined") {
        document.body.classList.add("embedded");
        // 네이티브에 준비 완료 알림
        try { CartoonBridge.onReady(); } catch (e) { /* noop */ }
    }
}

// ─────────────────────────────────────────────
// 스타일 선택
// ─────────────────────────────────────────────
function setupStylePicker() {
    stylePickerEl.addEventListener("click", (e) => {
        const btn = e.target.closest(".style-btn");
        if (!btn) return;
        const style = btn.dataset.style;
        if (style === currentStyle) return;

        // UI 업데이트
        stylePickerEl.querySelectorAll(".style-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentStyle = style;

        // 원본 있으면 즉시 재변환
        if (originalImageData) {
            cartoonize(originalImageData);
        }
    });
}

// ─────────────────────────────────────────────
// 파일 선택 (브라우저 개발용)
// ─────────────────────────────────────────────
function setupFileInput() {
    fileInputEl.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const dataUrl = await fileToDataUrl(file);
        processImage(dataUrl);
    });
}

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ─────────────────────────────────────────────
// 이미지 처리 진입점 (네이티브/브라우저 공통)
// ─────────────────────────────────────────────
async function processImage(dataUrl) {
    originalImageData = dataUrl;
    originalImgEl.src = dataUrl;
    cartoonImgEl.src = "";
    await cartoonize(dataUrl);
}

// ─────────────────────────────────────────────
// 카툰화 메인 로직
// ─────────────────────────────────────────────
async function cartoonize(dataUrl) {
    showLoading(true);
    setStatus(`${currentStyle} 스타일 처리 중...`);

    try {
        // 1. 모델 로드 (첫 회만 다운로드, 이후 캐시)
        const model = await loadModel(currentStyle);

        // 2. 이미지 → 텐서
        const img = await dataUrlToImageElement(dataUrl);
        const inputTensor = preprocessImage(img);

        // 3. 추론
        const t0 = performance.now();
        const outputTensor = model.execute(inputTensor);
        const dt = ((performance.now() - t0) / 1000).toFixed(1);

        // 4. 텐서 → 이미지
        const resultDataUrl = await tensorToDataUrl(outputTensor);

        // 5. 표시
        cartoonImgEl.src = resultDataUrl;
        setStatus(`완료 (${dt}초)`, "success");

        // 6. 정리
        tf.dispose([inputTensor, outputTensor]);

        // 7. 네이티브에 결과 전달
        if (typeof CartoonBridge !== "undefined") {
            try { CartoonBridge.onResult(resultDataUrl); } catch (e) { /* noop */ }
        }
    } catch (err) {
        console.error(err);
        setStatus(`오류: ${err.message}`, "error");
        if (typeof CartoonBridge !== "undefined") {
            try { CartoonBridge.onError(err.message); } catch (e) { /* noop */ }
        }
    } finally {
        showLoading(false);
    }
}

// ─────────────────────────────────────────────
// 모델 로드 (캐시)
// ─────────────────────────────────────────────
async function loadModel(style) {
    if (modelCache[style]) return modelCache[style];

    setStatus(`${style} 모델 다운로드 중...`);
    const url = MODEL_URLS[style];
    const model = await tf.loadGraphModel(url);
    modelCache[style] = model;
    return model;
}

// ─────────────────────────────────────────────
// 이미지 ↔ 텐서 변환
// ─────────────────────────────────────────────
function dataUrlToImageElement(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = dataUrl;
    });
}

/**
 * 전처리:
 *   - 짧은 변 기준 정사각 크롭 (얼굴 위주)
 *   - 최대 720px 리사이즈 (처리 부하 조절)
 *   - [0, 255] → [-1, 1] 정규화
 *   - 배치 차원 추가 [1, H, W, 3]
 */
function preprocessImage(img) {
    const maxDim = 720;
    // 짧은 변으로 정사각 크롭
    const size = Math.min(img.naturalWidth, img.naturalHeight);
    const cropSize = Math.min(size, maxDim);

    const canvas = document.createElement("canvas");
    canvas.width = cropSize;
    canvas.height = cropSize;
    const ctx = canvas.getContext("2d");

    // 중앙 크롭
    const sx = (img.naturalWidth - size) / 2;
    const sy = (img.naturalHeight - size) / 2;
    ctx.drawImage(img, sx, sy, size, size, 0, 0, cropSize, cropSize);

    return tf.tidy(() => {
        const raw = tf.browser.fromPixels(canvas);        // [H, W, 3], uint8
        const float = raw.toFloat();
        const normalized = float.div(127.5).sub(1);       // [-1, 1]
        return normalized.expandDims(0);                  // [1, H, W, 3]
    });
}

/**
 * 후처리:
 *   - [-1, 1] → [0, 255]
 *   - 텐서 → PNG DataUrl
 */
async function tensorToDataUrl(tensor) {
    // [1, H, W, 3] → [H, W, 3]
    const squeezed = tf.tidy(() => {
        const t = tensor.squeeze();
        return t.add(1).mul(127.5).clipByValue(0, 255).toInt();
    });

    const canvas = document.createElement("canvas");
    await tf.browser.toPixels(squeezed, canvas);
    squeezed.dispose();
    return canvas.toDataURL("image/jpeg", 0.92);
}

// ─────────────────────────────────────────────
// UI 헬퍼
// ─────────────────────────────────────────────
function setStatus(msg, type = "") {
    statusEl.textContent = msg;
    statusEl.className = "status" + (type ? " " + type : "");
}

function showLoading(show) {
    loadingEl.classList.toggle("hidden", !show);
}

// ─────────────────────────────────────────────
// 네이티브(안드로이드) 에서 호출하는 진입점
// ─────────────────────────────────────────────
// window.receiveImage(dataUrl)  — Android WebView 가 evaluateJavascript 로 호출
window.receiveImage = function(dataUrl) {
    processImage(dataUrl);
};

// window.setStyle(style)  — Android 에서 스타일 강제 지정
window.setStyle = function(style) {
    if (!MODEL_URLS[style]) return;
    currentStyle = style;
    stylePickerEl.querySelectorAll(".style-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.style === style);
    });
};
