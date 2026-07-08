# cartoon-demo

AnimeGANv2 기반 얼굴 → 애니메 캐릭터 변환 웹 데모.

안드로이드 네이티브 앱의 WebView 로 로드되어, 촬영한 사진을 브라우저 안에서 실제로 얼굴을 애니메 스타일로 변환합니다.

## 모델

**AnimeGANv2 face_paint_512** — 얼굴 자체 재구성

- 스타일 트랜스퍼 (색만 변경) 가 아니라 얼굴 구조를 애니메 캐릭터화
- 눈이 커지고, 이목구비 단순화, 애니메 특유 톤

모델 소스: [josephrocca/anime-gan-v2-web](https://github.com/josephrocca/anime-gan-v2-web)  
런타임: [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) (WASM 백엔드)  
파일 크기: 8.2MB (첫 회 다운로드 후 브라우저 캐시)

## 로컬 테스트

정적 파일이므로 아무 로컬 서버로 실행 가능:

```bash
cd web
python3 -m http.server 8080
# 브라우저에서 http://localhost:8080
```

또는 VS Code Live Server 확장 사용.

## GitHub Pages 배포

리포: `https://github.com/SongByungGyu/cartoon-demo`

### 최초 배포

```bash
cd web
git init
git add .
git commit -m "init: cartoon demo web page"
git branch -M main
git remote add origin https://github.com/SongByungGyu/cartoon-demo.git
git push -u origin main
```

### GitHub Pages 활성화

1. https://github.com/SongByungGyu/cartoon-demo/settings/pages
2. **Source**: `Deploy from a branch`
3. **Branch**: `main` / `/ (root)`
4. Save
5. 1-2분 후 배포 URL 활성화

### 배포 URL

`https://songbyunggyu.github.io/cartoon-demo/`

## 안드로이드 WebView 연동

### 네이티브 → 웹 (사진 전달)

```kotlin
webView.evaluateJavascript(
    "window.receiveImage('$base64DataUrl')",
    null
)
```

### 웹 → 네이티브 (결과 반환)

`CartoonBridge` 라는 이름의 JavaScript Interface 를 노출:

```kotlin
webView.addJavascriptInterface(object {
    @JavascriptInterface
    fun onReady() { /* 웹 준비 완료 */ }

    @JavascriptInterface
    fun onResult(dataUrl: String) { /* 카툰 결과 */ }

    @JavascriptInterface
    fun onError(message: String) { /* 오류 */ }
}, "CartoonBridge")
```

## 아키텍처 요약

```
[안드로이드 네이티브]
├── 카메라 · 얼굴 감지 · 크롭
├── WebView 로드 → https://songbyunggyu.github.io/cartoon-demo/
├── evaluateJavascript("receiveImage(base64)")
└── @JavascriptInterface CartoonBridge.onResult(...)

[WebView 안]
├── TensorFlow.js (CDN)
├── CartoonGAN 모델 (unpkg CDN)
└── WebGL/WebGPU 백엔드로 GPU 가속 추론
```

## 파일 구조

```
web/
├── index.html       메인 페이지
├── style.css        스타일
├── cartoonize.js    TF.js 로직 · 네이티브 브릿지
└── README.md        (이 파일)
```

모델은 로컬에 두지 않고 unpkg CDN 에서 로드 (리포 경량화).

## 라이선스 · 크레딧

- CartoonGAN 논문: Chen, Y., Lai, Y-K., Liu, Y-J. (CVPR 2018)
- 원본 TF 구현: [mnicnc404/CartoonGan-tensorflow](https://github.com/mnicnc404/CartoonGan-tensorflow)
- TF.js 변환본: [wangmengHB/local-tfjs-models](https://github.com/wangmengHB/local-tfjs-models)
