/**
 * Cartoon Demo — AnimeGANv2 (ONNX Runtime Web)
 *
 * josephrocca/anime-gan-v2-web 의 face_paint_512 모델을 활용하여
 * 실제로 얼굴을 애니메 캐릭터로 변형.
 *
 * 스타일 트랜스퍼(CartoonGAN) 와 달리 얼굴 자체가 재구성됨:
 *   - 눈이 커지고
 *   - 코·입이 애니메 스타일로 단순화
 *   - 머리카락·피부톤 애니메화
 *
 * 스펙:
 *   - 모델 파일: 8.2MB (josephrocca GitHub Pages CDN)
 *   - 입력: [1, 3, 512, 512] float32, RGB planar, [-1, 1] 정규화
 *   - 출력: 텐서 '940', 동일 형태
 *   - 백엔드: WASM (WebGL 은 int64 미지원)
 */

// ─────────────────────────────────────────────
// 상수 · 상태
// ─────────────────────────────────────────────
const MODEL_URL = "https://josephrocca.github.io/anime-gan-v2-web/anime-gan-v2.onnx";
const MODEL_SIZE = 512;

let session = null;              // ONNX InferenceSession
let originalImageData = null;    // dataUrl

// DOM
const statusEl = document.getElementById("status");
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

    // WASM 파일 로드 경로 (CDN)
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";

    setupFileInput();

    // 안드로이드 WebView 안에서 실행 중인지 감지
    if (typeof CartoonBridge !== "undefined") {
        document.body.classList.add("embedded");
    }

    // 모델은 첫 사용 시 로드 (초기 화면 로딩 빠르게)
    setStatus("사진을 선택하세요.", "success");

    // 네이티브에 준비 완료 알림
    if (typeof CartoonBridge !== "undefined") {
        try { CartoonBridge.onReady(); } catch (e) { /* noop */ }
    }
}

// ─────────────────────────────────────────────
// 파일 선택 (브라우저 개발용)
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

/**
 * File → EXIF orientation 반영된 DataUrl
 * createImageBitmap 옵션으로 브라우저가 회전 자동 적용.
 */
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
// 카툰화 메인 로직
// ─────────────────────────────────────────────
async function cartoonize(dataUrl) {
    showLoading(true);

    try {
        // 1. 모델 로드 (첫 회만)
        if (!session) {
            setStatus("AnimeGANv2 모델 다운로드 중 (8MB)...");
            session = await ort.InferenceSession.create(MODEL_URL, {
                executionProviders: ["wasm"]
            });
        }

        // 2. 이미지 → 텐서
        setStatus("전처리 중...");
        const img = await dataUrlToImageElement(dataUrl);
        const inputTensor = await preprocessImage(img);

        // 3. 추론
        setStatus("AI 처리 중... (5-15초)");
        const t0 = performance.now();
        const feeds = { "input.1": inputTensor };
        const results = await session.run(feeds);
        const dt = ((performance.now() - t0) / 1000).toFixed(1);

        // 4. 텐서 → 이미지
        setStatus("결과 생성 중...");
        const outputTensor = results["940"];
        const resultDataUrl = await tensorToDataUrl(outputTensor);

        // 5. 표시
        cartoonImgEl.src = resultDataUrl;
        setStatus(`완료 (${dt}초)`, "success");

        // 6. 네이티브에 결과 전달
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
 *   - 짧은 변 기준 중앙 정사각 크롭
 *   - 512x512 리사이즈
 *   - RGB planar 로 재배치 (HWC → CHW)
 *   - [0, 255] → [-1, 1] 정규화
 */
async function preprocessImage(img) {
    // 중앙 정사각 크롭 크기
    const size = Math.min(img.naturalWidth, img.naturalHeight);
    const sx = (img.naturalWidth - size) / 2;
    const sy = (img.naturalHeight - size) / 2;

    // 512x512 캔버스로 그림
    const canvas = document.createElement("canvas");
    canvas.width = MODEL_SIZE;
    canvas.height = MODEL_SIZE;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, sx, sy, size, size, 0, 0, MODEL_SIZE, MODEL_SIZE);

    // RGBA 픽셀 데이터 추출
    const imageData = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const rgba = imageData.data;

    // RGB planar (CHW) 형태로 재배치 + 정규화
    const planeSize = MODEL_SIZE * MODEL_SIZE;
    const chw = new Float32Array(planeSize * 3);

    for (let i = 0; i < planeSize; i++) {
        const r = rgba[i * 4 + 0];
        const g = rgba[i * 4 + 1];
        const b = rgba[i * 4 + 2];
        chw[i]                  = (r / 255) * 2 - 1;  // R plane
        chw[planeSize + i]      = (g / 255) * 2 - 1;  // G plane
        chw[planeSize * 2 + i]  = (b / 255) * 2 - 1;  // B plane
    }

    return new ort.Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

/**
 * 후처리:
 *   - [-1, 1] → [0, 255]
 *   - RGB planar (CHW) → RGBA interleaved (HWC + alpha)
 *   - 캔버스 → JPEG DataUrl
 */
async function tensorToDataUrl(tensor) {
    const data = tensor.data;              // Float32Array, planar
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
