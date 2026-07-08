/**
 * Cartoon Demo — AnimeGANv2 (4 Styles)
 *
 * bryandlee/animegan2-pytorch 의 4가지 사전학습 모델을 자체 변환한 ONNX.
 * 우리 GitHub Pages 에서 직접 서빙.
 *
 * 스타일:
 *   - celeba_distill   : CelebA 학습, 부드러운 톤. **성별·정체성 보존 우수**
 *   - face_paint_512_v1: 클래식 애니메 스타일 (v2 이전)
 *   - face_paint_512_v2: 강한 애니메 (여성화 편향 있음)
 *   - paprika         : 파프리카 영화 몽환적 스타일
 *
 * 스펙:
 *   - 입력: [1, 3, 512, 512] float32, RGB planar, [-1, 1]
 *   - 출력: [1, 3, 512, 512] float32, RGB planar, [-1, 1]
 *   - 텐서명: input, output
 *   - 백엔드: WASM
 */

// ─────────────────────────────────────────────
// 모델 URL · 스타일 메타
// ─────────────────────────────────────────────
const STYLES = {
    celeba_distill: {
        url: "./models/celeba_distill.onnx",
        label: "부드러움 (성별 유지)"
    },
    face_paint_512_v1: {
        url: "./models/face_paint_512_v1.onnx",
        label: "클래식 애니메"
    },
    face_paint_512_v2: {
        url: "./models/face_paint_512_v2.onnx",
        label: "강한 애니메"
    },
    paprika: {
        url: "./models/paprika.onnx",
        label: "파프리카 몽환"
    }
};

const MODEL_SIZE = 512;

// ─────────────────────────────────────────────
// 상태
// ─────────────────────────────────────────────
const sessions = {};                       // 스타일별 로드된 세션 캐시
let currentStyle = "celeba_distill";       // 기본: 성별 편향 완화
let originalImageData = null;

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
    setStatus("ONNX Runtime 준비 중...");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

    setupStylePicker();
    setupFileInput();

    if (typeof CartoonBridge !== "undefined") {
        document.body.classList.add("embedded");
    }

    setStatus("사진을 선택하세요.", "success");

    if (typeof CartoonBridge !== "undefined") {
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

        stylePickerEl.querySelectorAll(".style-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentStyle = style;

        if (originalImageData) {
            cartoonize(originalImageData);
        }
    });
}

// ─────────────────────────────────────────────
// 파일 선택 (개발용)
// ─────────────────────────────────────────────
function setupFileInput() {
    fileInputEl.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const dataUrl = await fileToOrientedDataUrl(file);
            processImage(dataUrl);
        } catch (err) {
            console.error(err);
            setStatus(`파일 처리 실패: ${err.message}`, "error");
        }
    });
}

async function fileToOrientedDataUrl(file) {
    let bitmap;
    try {
        bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
        bitmap = await createImageBitmap(file);
    }

    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close?.();

    return canvas.toDataURL("image/jpeg", 0.95);
}

// ─────────────────────────────────────────────
// 이미지 처리 진입점
// ─────────────────────────────────────────────
async function processImage(dataUrl) {
    originalImageData = dataUrl;
    originalImgEl.src = dataUrl;
    cartoonImgEl.src = "";
    await cartoonize(dataUrl);
}

// ─────────────────────────────────────────────
// 카툰화 메인
// ─────────────────────────────────────────────
async function cartoonize(dataUrl) {
    showLoading(true);

    try {
        const session = await loadSession(currentStyle);

        setStatus("전처리 중...");
        const img = await dataUrlToImageElement(dataUrl);
        const inputTensor = preprocessImage(img);

        setStatus(`${STYLES[currentStyle].label} 처리 중...`);
        const t0 = performance.now();
        const results = await session.run({ input: inputTensor });
        const dt = ((performance.now() - t0) / 1000).toFixed(1);

        setStatus("결과 생성 중...");
        const resultDataUrl = tensorToDataUrl(results.output);

        cartoonImgEl.src = resultDataUrl;
        setStatus(`완료 (${dt}초)`, "success");

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
// 세션 로드 (스타일별 캐시)
// ─────────────────────────────────────────────
async function loadSession(style) {
    if (sessions[style]) return sessions[style];

    setStatus(`${STYLES[style].label} 모델 다운로드 중 (8MB)...`);
    const session = await ort.InferenceSession.create(STYLES[style].url, {
        executionProviders: ["wasm"]
    });
    sessions[style] = session;
    return session;
}

// ─────────────────────────────────────────────
// 이미지 ↔ 텐서
// ─────────────────────────────────────────────
function dataUrlToImageElement(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = dataUrl;
    });
}

function preprocessImage(img) {
    // 중앙 정사각 크롭 → 512x512
    const size = Math.min(img.naturalWidth, img.naturalHeight);
    const sx = (img.naturalWidth - size) / 2;
    const sy = (img.naturalHeight - size) / 2;

    const canvas = document.createElement("canvas");
    canvas.width = MODEL_SIZE;
    canvas.height = MODEL_SIZE;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, sx, sy, size, size, 0, 0, MODEL_SIZE, MODEL_SIZE);

    const imageData = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const rgba = imageData.data;

    // RGB planar + 정규화 [-1, 1]
    const planeSize = MODEL_SIZE * MODEL_SIZE;
    const chw = new Float32Array(planeSize * 3);

    for (let i = 0; i < planeSize; i++) {
        chw[i]                  = (rgba[i * 4 + 0] / 255) * 2 - 1;
        chw[planeSize + i]      = (rgba[i * 4 + 1] / 255) * 2 - 1;
        chw[planeSize * 2 + i]  = (rgba[i * 4 + 2] / 255) * 2 - 1;
    }

    return new ort.Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function tensorToDataUrl(tensor) {
    const data = tensor.data;
    const planeSize = MODEL_SIZE * MODEL_SIZE;
    const rgba = new Uint8ClampedArray(planeSize * 4);

    for (let i = 0; i < planeSize; i++) {
        const r = data[i] * 0.5 + 0.5;
        const g = data[planeSize + i] * 0.5 + 0.5;
        const b = data[planeSize * 2 + i] * 0.5 + 0.5;

        rgba[i * 4 + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
        rgba[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
        rgba[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
        rgba[i * 4 + 3] = 255;
    }

    const imageData = new ImageData(rgba, MODEL_SIZE, MODEL_SIZE);
    const canvas = document.createElement("canvas");
    canvas.width = MODEL_SIZE;
    canvas.height = MODEL_SIZE;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
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
// 네이티브(안드로이드) 진입점
// ─────────────────────────────────────────────
window.receiveImage = function(dataUrl) {
    processImage(dataUrl);
};

window.setStyle = function(style) {
    if (!STYLES[style]) return;
    currentStyle = style;
    stylePickerEl.querySelectorAll(".style-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.style === style);
    });
};
