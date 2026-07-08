/**
 * Cartoon Demo — Hybrid: FLUX Kontext (HD) + AnimeGANv2 (Local)
 *
 * HD 모드: HuggingFace Inference Providers 를 통해 fal-ai 에서 FLUX Kontext 실행
 *   - 얼굴·정체성 보존 우수
 *   - 다양한 스타일 프롬프트로 변환
 *   - 월 80장 무료 (HF $0.10 크레딧)
 *   - 5-10초 소요
 *
 * 로컬 모드: 브라우저 안에서 AnimeGANv2 4스타일 실행
 *   - 완전 무료 · 무제한 · 오프라인
 *   - 품질 제한적
 *
 * ⚠️ 보안 참고
 *   현재 데모 단계라 HF 토큰이 JS 에 직접 노출됨.
 *   사용자 계정에 결제수단 없어 금전 손실 위험 없으나,
 *   프로덕션 전환 시 Cloudflare Worker 등 프록시로 이관 필요.
 */

// ─────────────────────────────────────────────
// 설정
// ─────────────────────────────────────────────

// HF 토큰은 사용자 브라우저 localStorage 에 저장. 리포에 안 들어감.
// HD 모드 첫 사용 시 입력 요구. 이후 이 브라우저에서만 자동 사용.
const HF_ENDPOINT = "https://router.huggingface.co/fal-ai/fal-ai/flux-kontext/dev";
const HF_TOKEN_KEY = "hf_token";

function getHfToken() {
    return localStorage.getItem(HF_TOKEN_KEY);
}

function setHfToken(token) {
    if (token && token.startsWith("hf_")) {
        localStorage.setItem(HF_TOKEN_KEY, token);
        return true;
    }
    return false;
}

function clearHfToken() {
    localStorage.removeItem(HF_TOKEN_KEY);
}

async function ensureHfToken() {
    let token = getHfToken();
    if (token) return token;

    // 프롬프트 UI · 사용자에게 토큰 요청
    const input = prompt(
        "HD 모드 사용을 위해 HuggingFace 토큰 입력 (hf_로 시작)\n\n" +
        "https://huggingface.co/settings/tokens 에서 생성 가능 (Read 권한).\n" +
        "이 브라우저 localStorage 에만 저장되며 서버·리포로 나가지 않습니다."
    );
    if (!input) throw new Error("토큰 입력 필요");
    if (!setHfToken(input.trim())) {
        throw new Error("잘못된 토큰 형식 (hf_ 로 시작해야 함)");
    }
    return getHfToken();
}

// HD 스타일 (FLUX Kontext) — 각 스타일별 최적화된 프롬프트
const HD_STYLES = {
    anime: {
        label: "애니메 캐릭터",
        prompt: "Convert this photo into a beautiful anime character illustration, expressive large eyes, clean line art, vibrant colors, preserve face features and identity, high quality anime style"
    },
    pixar: {
        label: "픽사 3D",
        prompt: "Transform this photo into a Pixar Disney 3D animated movie character, cute expressive, cinematic lighting, preserve face features and identity, high quality 3D render"
    },
    ghibli: {
        label: "지브리",
        prompt: "Transform this photo into Studio Ghibli Miyazaki Hayao anime style, soft watercolor, warm lighting, preserve face features and identity, high quality Ghibli illustration"
    },
    disney: {
        label: "디즈니 2D",
        prompt: "Transform this photo into a classic Disney 2D animated movie character, colorful, expressive, preserve face features and identity, high quality Disney illustration"
    },
    webtoon: {
        label: "웹툰",
        prompt: "Convert this photo into Korean webtoon manhwa style illustration, clean line art, soft coloring, preserve face features and identity, high quality webtoon style"
    },
    semirealistic: {
        label: "실사 카툰",
        prompt: "Transform this photo into a semi-realistic stylized cartoon illustration, painterly, preserve face features and identity, high quality digital illustration"
    }
};

// 로컬 스타일 (AnimeGANv2)
const LOCAL_STYLES = {
    celeba_distill:     { label: "부드러움",   url: "./models/celeba_distill.onnx" },
    face_paint_512_v1:  { label: "애니 v1",    url: "./models/face_paint_512_v1.onnx" },
    face_paint_512_v2:  { label: "애니 v2",    url: "./models/face_paint_512_v2.onnx" },
    paprika:            { label: "파프리카",   url: "./models/paprika.onnx" }
};

const MODEL_SIZE = 512;

// ─────────────────────────────────────────────
// 상태
// ─────────────────────────────────────────────
const localSessions = {};
let currentMode = "hd";        // 'hd' or 'local'
let currentStyle = "anime";    // key of HD_STYLES or LOCAL_STYLES
let originalImageData = null;

// DOM
const statusEl = document.getElementById("status");
const stylePickerHdEl = document.getElementById("stylePickerHd");
const stylePickerLocalEl = document.getElementById("stylePickerLocal");
const originalImgEl = document.getElementById("original");
const cartoonImgEl = document.getElementById("cartoon");
const loadingEl = document.getElementById("loading");
const fileInputEl = document.getElementById("fileInput");

// ─────────────────────────────────────────────
// 초기화
// ─────────────────────────────────────────────
init();

async function init() {
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
// 스타일 선택 · 두 피커 통합 관리
// ─────────────────────────────────────────────
function setupStylePicker() {
    [stylePickerHdEl, stylePickerLocalEl].forEach(picker => {
        picker.addEventListener("click", (e) => {
            const btn = e.target.closest(".style-btn");
            if (!btn) return;
            const mode = btn.dataset.mode;
            const style = btn.dataset.style;
            if (mode === currentMode && style === currentStyle) return;

            // 모든 버튼 active 해제 → 클릭한 것만 active
            document.querySelectorAll(".style-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            currentMode = mode;
            currentStyle = style;

            if (originalImageData) {
                cartoonize(originalImageData);
            }
        });
    });
}

// ─────────────────────────────────────────────
// 파일 선택
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
// 카툰화 라우팅
// ─────────────────────────────────────────────
async function cartoonize(dataUrl) {
    showLoading(true);
    try {
        let resultDataUrl;
        if (currentMode === "hd") {
            resultDataUrl = await cartoonizeHd(dataUrl);
        } else {
            resultDataUrl = await cartoonizeLocal(dataUrl);
        }
        cartoonImgEl.src = resultDataUrl;

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
// HD 모드: FLUX Kontext via HF Inference Providers
// ─────────────────────────────────────────────
async function cartoonizeHd(dataUrl) {
    const style = HD_STYLES[currentStyle];
    if (!style) throw new Error("알 수 없는 HD 스타일");

    const token = await ensureHfToken();

    setStatus(`${style.label} 처리 중... (5-15초, HD)`);

    // 원본 크기 · 화질 조절 (전송량 최소화)
    const inputDataUrl = await resizeToMaxDim(dataUrl, 768);

    const t0 = performance.now();
    const response = await fetch(HF_ENDPOINT, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            prompt: style.prompt,
            image_url: inputDataUrl
        })
    });

    if (!response.ok) {
        const errBody = await response.text();
        throw new Error(`HF API ${response.status}: ${errBody.slice(0, 200)}`);
    }

    const data = await response.json();
    if (!data.images || !data.images[0]) {
        throw new Error("응답에 이미지 없음: " + JSON.stringify(data).slice(0, 200));
    }

    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`${style.label} 완료 (${dt}초 · HD)`, "success");

    // 결과 URL 을 base64 dataUrl 로 변환 (안드로이드 WebView 저장·공유용)
    return await urlToDataUrl(data.images[0].url);
}

async function resizeToMaxDim(dataUrl, maxDim) {
    const img = await dataUrlToImageElement(dataUrl);
    const long = Math.max(img.naturalWidth, img.naturalHeight);
    if (long <= maxDim) return dataUrl;

    const ratio = maxDim / long;
    const w = Math.round(img.naturalWidth * ratio);
    const h = Math.round(img.naturalHeight * ratio);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(img, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.92);
}

async function urlToDataUrl(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

// ─────────────────────────────────────────────
// 로컬 모드: AnimeGANv2 ONNX
// ─────────────────────────────────────────────
async function cartoonizeLocal(dataUrl) {
    const style = LOCAL_STYLES[currentStyle];
    if (!style) throw new Error("알 수 없는 로컬 스타일");

    // 모델 로드
    if (!localSessions[currentStyle]) {
        setStatus(`${style.label} 모델 다운로드 중 (8MB)...`);
        localSessions[currentStyle] = await ort.InferenceSession.create(style.url, {
            executionProviders: ["wasm"]
        });
    }

    setStatus("전처리 중...");
    const img = await dataUrlToImageElement(dataUrl);
    const inputTensor = preprocessLocal(img);

    setStatus(`${style.label} 처리 중... (로컬)`);
    const t0 = performance.now();
    const results = await localSessions[currentStyle].run({ input: inputTensor });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);

    const outputDataUrl = tensorToDataUrl(results.output);
    setStatus(`${style.label} 완료 (${dt}초 · 로컬)`, "success");
    return outputDataUrl;
}

function dataUrlToImageElement(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = dataUrl;
    });
}

function preprocessLocal(img) {
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
    canvas.getContext("2d").putImageData(imageData, 0, 0);
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

window.setMode = function(mode, style) {
    if (mode === "hd" && HD_STYLES[style]) {
        currentMode = "hd";
        currentStyle = style;
    } else if (mode === "local" && LOCAL_STYLES[style]) {
        currentMode = "local";
        currentStyle = style;
    }
    document.querySelectorAll(".style-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.mode === currentMode && b.dataset.style === currentStyle);
    });
};
