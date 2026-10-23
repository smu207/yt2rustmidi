/*
 * yt2midi-web 변환 파이프라인 (브라우저 전용)
 *
 *   오디오 파일 → (Demucs, ONNX) 보컬/드럼/베이스/기타 악기 분리
 *              → (Basic Pitch, TF.js) 음정 악기를 음표로 변환
 *              → 드럼은 주파수 대역별 타격 검출
 *              → 템포 추정 → 편집기용 곡 데이터(ticks)
 *
 * esbuild로 묶어서 app.bundle.js 하나로 만들고, 전역 YT2MIDI 로 노출합니다.
 * onnxruntime-web 은 index.html 에서 CDN 스크립트로 불러온 window.ort 를 사용합니다.
 */
import { DemucsProcessor, CONSTANTS } from 'demucs-web';
import * as tf from '@tensorflow/tfjs';
import {
  BasicPitch,
  outputToNotesPoly,
  addPitchBendsToNoteEvents,
  noteFramesToTime,
} from '@spotify/basic-pitch';

export const ORT_VERSION = '1.20.1';
export const DEMUCS_MODEL_URL = CONSTANTS.DEFAULT_MODEL_URL; // 약 172MB, 첫 실행 때 한 번 내려받음
// 6트랙 모델 (기타·피아노 분리, 실험적). 입력 mix [1,2,343980] → 출력 stems [1,6,2,343980]
export const DEMUCS6_MODEL_URL = 'https://huggingface.co/adowu/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx'; // 약 136MB
const SIX_ORDER = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
const SEG = 343980;
export const STEMS = ['vocals', 'drums', 'bass', 'other'];
export const STEMS6 = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];
export const STEM_LABELS = { vocals: '보컬', drums: '드럼', bass: '베이스', guitar: '기타', piano: '피아노', other: '기타 악기' };

// 악기별 설정 (GM 프로그램 번호, 음역 Hz, 단선율 여부, Basic Pitch 임계값)
export const STEM_CONFIG = {
  vocals: { name: 'Vocals', program: 53, minFreq: 80, maxFreq: 1200, mono: true, onset: 0.5, frame: 0.3, minLenMs: 100 },
  bass: { name: 'Bass', program: 33, minFreq: 30, maxFreq: 400, mono: true, onset: 0.5, frame: 0.3, minLenMs: 80 },
  guitar: { name: 'Guitar', program: 27, minFreq: 70, maxFreq: 1500, mono: false, onset: 0.5, frame: 0.3, minLenMs: 80 },
  piano: { name: 'Piano', program: 0, minFreq: 27, maxFreq: 4200, mono: false, onset: 0.5, frame: 0.3, minLenMs: 60 },
  other: { name: 'Other', program: 0, minFreq: 50, maxFreq: 3000, mono: false, onset: 0.6, frame: 0.4, minLenMs: 100 },
};
const DRUM_BANDS = [
  // [GM 노트, 이름, 필터 체인]
  [36, 'Kick', [['lowpass', 150], ['lowpass', 150]]],
  [38, 'Snare', [['highpass', 180], ['lowpass', 2500]]],
  [42, 'Hi-hat', [['highpass', 6000], ['highpass', 6000]]],
];
const SR = 44100;
const BP_SR = 22050;
const BP_FPS = Math.floor(BP_SR / 256); // Basic Pitch 프레임 속도 (86fps)
const SILENCE_RMS = 1e-3;
const PPQ = 480;
const CACHE_NAME = 'yt2midi-models-v1';

/* ---------------- 공통 도우미 ---------------- */
class Cancelled extends Error { constructor() { super('취소했습니다'); this.cancelled = true; } }
function checkAbort(signal) { if (signal && signal.aborted) throw new Cancelled(); }
const nextFrame = () => new Promise(r => setTimeout(r, 0));

async function fetchWithCache(url, onProgress, signal) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch (_) { /* 캐시 사용 불가 */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { onProgress(1, 1, true); return await hit.arrayBuffer(); }
  }
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error('분리 모델을 내려받지 못했습니다 (HTTP ' + res.status + '). 인터넷 연결을 확인해 주세요.');
  const total = parseInt(res.headers.get('Content-Length') || '0', 10);
  let buf;
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total || loaded, false);
    }
    const all = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    buf = all.buffer;
  } else {
    buf = await res.arrayBuffer();
  }
  if (cache) {
    try { await cache.put(url, new Response(buf.slice(0), { headers: { 'Content-Type': 'application/octet-stream' } })); }
    catch (_) { /* 저장 공간 부족 등: 다음에 다시 내려받음 */ }
  }
  return buf;
}

export async function clearModelCache() {
  try { return await caches.delete(CACHE_NAME); } catch (_) { return false; }
}

export async function decodeFile(file) {
  const ab = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  try {
    return await new Promise((resolve, reject) => ctx.decodeAudioData(ab, resolve, () => reject(new Error('이 오디오 파일을 읽을 수 없습니다. mp3, wav, m4a, ogg, flac 파일을 사용해 주세요.'))));
  } finally {
    try { ctx.close(); } catch (_) {}
  }
}

async function render(buffer, sampleRate, channels, startSec, durSec, filters) {
  const len = Math.max(1, Math.ceil(durSec * sampleRate));
  const oac = new OfflineAudioContext(channels, len, sampleRate);
  const src = oac.createBufferSource();
  src.buffer = buffer;
  let node = src;
  (filters || []).forEach(([type, freq]) => {
    const f = oac.createBiquadFilter();
    f.type = type; f.frequency.value = freq; f.Q.value = 0.707;
    node.connect(f); node = f;
  });
  node.connect(oac.destination);
  src.start(0, startSec, durSec);
  return await oac.startRendering();
}

function toAudioBuffer(left, right, sampleRate) {
  const b = new AudioBuffer({ length: left.length, numberOfChannels: 2, sampleRate });
  b.copyToChannel(left, 0);
  b.copyToChannel(right, 1);
  return b;
}

function rms(stereo) {
  let s = 0;
  const n = stereo.left.length;
  const step = Math.max(1, Math.floor(n / 400000));
  let c = 0;
  for (let i = 0; i < n; i += step) { s += stereo.left[i] * stereo.left[i] + stereo.right[i] * stereo.right[i]; c += 2; }
  return Math.sqrt(s / Math.max(1, c));
}

export function encodeWav(left, right, sampleRate) {
  const n = left.length;
  const buf = new ArrayBuffer(44 + n * 4);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); w(8, 'WAVE');
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 4, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const l = Math.max(-1, Math.min(1, left[i])), r = Math.max(-1, Math.min(1, right[i]));
    v.setInt16(o, l < 0 ? l * 0x8000 : l * 0x7fff, true);
    v.setInt16(o + 2, r < 0 ? r * 0x8000 : r * 0x7fff, true);
    o += 4;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/* ---------------- 1) 음원 분리 ---------------- */
async function pickProviders(device) {
  if (device === 'wasm') return ['wasm'];
  try {
    if (navigator.gpu && await navigator.gpu.requestAdapter()) return ['webgpu', 'wasm'];
  } catch (_) {}
  return ['wasm'];
}

function setupOrt() {
  const ort = window.ort;
  if (!ort) throw new Error('ONNX Runtime 을 불러오지 못했습니다. 인터넷 연결을 확인하고 페이지를 새로고침해 주세요.');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/';
  if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;
  return ort;
}
const mb = n => (n / 1048576).toFixed(0);
const modelProgress = ui => (loaded, total, cached) =>
  ui.progress('model', total ? loaded / total : 0, cached ? '저장된 모델 사용' : mb(loaded) + ' / ' + mb(total) + 'MB');

/* 6트랙: 모델을 직접 실행하고 구간을 겹쳐 이어 붙임 (overlap-add) */
async function separate6(stereoBuffer, opts, ui) {
  const ort = setupOrt();
  ui.stage('model', '6트랙 분리 모델 준비 중');
  const buf = await fetchWithCache(DEMUCS6_MODEL_URL, modelProgress(ui), opts.signal);
  checkAbort(opts.signal);

  let providers = await pickProviders(opts.device);
  const create = eps => ort.InferenceSession.create(new Uint8Array(buf), {
    executionProviders: eps, graphOptimizationLevel: 'basic', enableCpuMemArena: false, enableMemPattern: false,
  });
  let session;
  try { session = await create(providers); }
  catch (e) { if (providers[0] !== 'webgpu') throw e; providers = ['wasm']; session = await create(providers); }
  ui.log((providers[0] === 'webgpu' ? 'GPU(WebGPU)' : 'CPU(WASM)') + '로 6트랙 분리를 시작합니다.');

  const inName = session.inputNames.includes('mix') ? 'mix' : session.inputNames[0];
  const outName = session.outputNames.includes('stems') ? 'stems' : session.outputNames[0];
  const L = stereoBuffer.getChannelData(0), R = stereoBuffer.getChannelData(1);
  const total = L.length;
  const stride = Math.floor(SEG * 0.75);
  const nSeg = total <= SEG ? 1 : Math.ceil((total - SEG) / stride) + 1;
  const outs = SIX_ORDER.map(() => ({ left: new Float32Array(total), right: new Float32Array(total) }));
  const weights = new Float32Array(total);
  const fadeLen = stride * 0.5;

  ui.stage('separate', '악기 분리 중 (6트랙)');
  for (let k = 0, start = 0; k < nSeg; k++, start += stride) {
    checkAbort(opts.signal);
    const len = Math.min(SEG, total - start);
    if (len <= 0) break;
    const input = new Float32Array(2 * SEG);
    input.set(L.subarray(start, start + len), 0);
    input.set(R.subarray(start, start + len), SEG);
    const feeds = {}; feeds[inName] = new ort.Tensor('float32', input, [1, 2, SEG]);
    let res;
    try { res = await session.run(feeds); }
    catch (e) {
      if (providers[0] !== 'webgpu') throw e;
      ui.log('GPU 실행에 실패해서 CPU로 바꿉니다: ' + e.message);
      try { await session.release(); } catch (_) {}
      providers = ['wasm']; session = await create(providers);
      res = await session.run(feeds);
    }
    const out = res[outName];
    const data = out.data;
    if (!(data instanceof Float32Array) || data.length !== 6 * 2 * SEG) throw new Error('6트랙 모델 출력 형식이 예상과 다릅니다 (' + (out.dims || []).join('x') + ')');
    for (let i = 0; i < len; i++) {
      const w = Math.min(1, (i + 1) / fadeLen, (len - i) / fadeLen);
      weights[start + i] += w;
    }
    for (let t = 0; t < 6; t++) {
      const oL = (t * 2) * SEG, oR = (t * 2 + 1) * SEG, dst = outs[t];
      for (let i = 0; i < len; i++) {
        const w = Math.min(1, (i + 1) / fadeLen, (len - i) / fadeLen);
        dst.left[start + i] += data[oL + i] * w;
        dst.right[start + i] += data[oR + i] * w;
      }
    }
    if (out.dispose) out.dispose();
    ui.progress('separate', (k + 1) / nSeg, (k + 1) + ' / ' + nSeg + ' 구간');
    await nextFrame();
  }
  for (const o of outs) for (let i = 0; i < total; i++) if (weights[i] > 0) { o.left[i] /= weights[i]; o.right[i] /= weights[i]; }
  try { await session.release(); } catch (_) {}
  const result = {};
  SIX_ORDER.forEach((name, i) => { result[name] = outs[i]; });
  return result;
}

async function separate(stereoBuffer, opts, ui) {
  if (opts.model === '6') {
    try {
      return { stems: await separate6(stereoBuffer, opts, ui), mode: '6' };
    } catch (e) {
      if (e && e.cancelled) throw e;
      ui.log('6트랙 분리에 실패해서 4트랙으로 바꿉니다: ' + ((e && e.message) || e));
      ui.reset && ui.reset(['model', 'separate']);
    }
  }
  return { stems: await separate4(stereoBuffer, opts, ui), mode: '4' };
}

async function separate4(stereoBuffer, opts, ui) {
  const ort = setupOrt();

  ui.stage('model', '분리 모델 준비 중');
  const modelBuf = await fetchWithCache(DEMUCS_MODEL_URL, modelProgress(ui), opts.signal);
  checkAbort(opts.signal);

  let providers = await pickProviders(opts.device);
  const make = eps => new DemucsProcessor({
    ort,
    sessionOptions: { executionProviders: eps, enableCpuMemArena: false, enableMemPattern: false },
    onProgress: p => {
      checkAbort(opts.signal);
      ui.progress('separate', p.progress, p.currentSegment + ' / ' + p.totalSegments + ' 구간');
    },
  });
  let proc = make(providers);
  try {
    await proc.loadModel(modelBuf.slice(0));
  } catch (e) {
    if (providers[0] !== 'webgpu') throw e;
    providers = ['wasm'];
    proc = make(providers);
    await proc.loadModel(modelBuf.slice(0));
  }
  ui.log(providers[0] === 'webgpu' ? 'GPU(WebGPU)로 분리합니다.' : 'CPU(WASM)로 분리합니다. 곡 길이에 따라 수 분~십여 분 걸릴 수 있어요.');

  ui.stage('separate', '악기 분리 중');
  await nextFrame();
  const left = stereoBuffer.getChannelData(0), right = stereoBuffer.getChannelData(1);
  let result;
  try {
    result = await proc.separate(left, right);
  } catch (e) {
    if (e.cancelled || providers[0] !== 'webgpu') throw e;
    ui.log('GPU 분리에 실패해서 CPU로 다시 시도합니다: ' + e.message);
    proc = make(['wasm']);
    await proc.loadModel(modelBuf.slice(0));
    result = await proc.separate(left, right);
  }
  try { if (proc.session && proc.session.release) await proc.session.release(); } catch (_) {}
  return result; // { drums, bass, other, vocals } 각각 {left, right}
}

/* ---------------- 2-a) 음정 악기 → 음표 ---------------- */
let basicPitch = null;
function getBasicPitch(modelUrl) {
  if (!basicPitch) basicPitch = new BasicPitch(modelUrl);
  return basicPitch;
}

function makeMonophonic(notes, minDur) {
  notes.sort((a, b) => a.start - b.start || b.vel - a.vel);
  const out = [];
  for (const n of notes) {
    const prev = out[out.length - 1];
    if (prev && n.start < prev.end) {
      if (Math.abs(n.start - prev.start) < 0.03) {
        if (n.vel > prev.vel) out[out.length - 1] = n;
        continue;
      }
      prev.end = n.start;
    }
    out.push(n);
  }
  return out.filter(n => n.end - n.start >= (minDur || 0.05));
}

async function transcribePitched(stereo, cfg, modelUrl, onP, signal) {
  const ab = toAudioBuffer(stereo.left, stereo.right, SR);
  const mono = await render(ab, BP_SR, 1, 0, ab.duration);
  const data = mono.getChannelData(0);
  const frames = [], onsets = [], contours = [];
  await getBasicPitch(modelUrl).evaluateModel(data, (f, o, c) => {
    frames.push(...f); onsets.push(...o); contours.push(...c);
  }, p => { checkAbort(signal); onP(p); });
  const minLenFrames = Math.max(1, Math.round(cfg.minLenMs / 1000 * BP_FPS));
  const events = noteFramesToTime(addPitchBendsToNoteEvents(
    contours,
    outputToNotesPoly(frames, onsets, cfg.onset, cfg.frame, minLenFrames, true, cfg.maxFreq, cfg.minFreq, true),
  ));
  let notes = events.map(e => ({
    pitch: e.pitchMidi,
    start: e.startTimeSeconds,
    end: e.startTimeSeconds + e.durationSeconds,
    vel: Math.max(1, Math.min(127, Math.round(e.amplitude * 127))),
  }));
  if (cfg.mono) notes = makeMonophonic(notes);
  return notes;
}

/* ---------------- 2-b) 드럼 → 음표 ---------------- */
function onsetEnvelope(data, hop, win) {
  const n = Math.floor((data.length - win) / hop) + 1;
  const env = new Float32Array(Math.max(0, n));
  let prev = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * hop;
    for (let j = 0; j < win; j++) { const x = data[o + j]; s += x * x; }
    const e = Math.sqrt(s / win);
    env[i] = Math.max(0, e - prev);
    prev = e;
  }
  return env;
}

function pickPeaks(env, fps, deltaRatio, waitSec) {
  let max = 0;
  for (let i = 0; i < env.length; i++) if (env[i] > max) max = env[i];
  if (max <= 0) return [];
  const delta = deltaRatio * max;
  const half = Math.round(fps * 0.25);
  const wait = Math.max(1, Math.round(waitSec * fps));
  // 이동 평균 (적응 임계값)
  const pre = new Float64Array(env.length + 1);
  for (let i = 0; i < env.length; i++) pre[i + 1] = pre[i] + env[i];
  const peaks = [];
  let last = -Infinity;
  for (let i = 0; i < env.length; i++) {
    const v = env[i];
    if ((i > 0 && v < env[i - 1]) || (i + 1 < env.length && v < env[i + 1])) continue;
    let isMax = true;
    for (let k = Math.max(0, i - 3); k <= Math.min(env.length - 1, i + 3); k++) if (env[k] > v) { isMax = false; break; }
    if (!isMax) continue;
    const a = Math.max(0, i - half), b = Math.min(env.length, i + half + 1);
    const mean = (pre[b] - pre[a]) / (b - a);
    if (v < mean + delta) continue;
    if (i - last < wait) continue;
    peaks.push(i);
    last = i;
  }
  return peaks;
}

async function transcribeDrums(stereo, ui) {
  const ab = toAudioBuffer(stereo.left, stereo.right, SR);
  const hop = 256, win = 1024, fps = SR / hop;
  const notes = [];
  for (const [pitch, label, chain] of DRUM_BANDS) {
    const band = await render(ab, SR, 1, 0, ab.duration, chain);
    const env = onsetEnvelope(band.getChannelData(0), hop, win);
    const peaks = pickPeaks(env, fps, 0.07, 0.05);
    const vals = peaks.map(i => env[i]).sort((a, b) => a - b);
    const p95 = vals.length ? vals[Math.floor(vals.length * 0.95)] || vals[vals.length - 1] : 1;
    peaks.forEach(i => {
      const t = (i * hop + win / 2) / SR;
      const vel = Math.max(30, Math.min(127, Math.round(40 + 87 * env[i] / (p95 + 1e-9))));
      notes.push({ pitch, start: t, end: t + 0.1, vel });
    });
    ui.log('드럼 ' + label + ': ' + peaks.length + '타');
    await nextFrame();
  }
  notes.sort((a, b) => a.start - b.start);
  return notes;
}

/* ---------------- 3) 템포 추정 ---------------- */
export function estimateTempo(monoData, sampleRate) {
  const hop = 512, win = 1024, fps = sampleRate / hop;
  const maxLen = Math.min(monoData.length, sampleRate * 120);
  const env = onsetEnvelope(monoData.subarray(0, maxLen), hop, win);
  if (env.length < fps * 4) return 120;
  let mean = 0;
  for (let i = 0; i < env.length; i++) mean += env[i];
  mean /= env.length;
  const x = Array.from(env, v => v - mean);
  let best = 120, bestScore = -Infinity;
  for (let bpm = 60; bpm <= 180; bpm += 0.5) {
    const lag = 60 * fps / bpm;
    const l0 = Math.floor(lag), fr = lag - l0;
    let s = 0;
    for (let i = 0; i + l0 + 1 < x.length; i++) s += x[i] * (x[i + l0] * (1 - fr) + x[i + l0 + 1] * fr);
    s /= (x.length - l0);
    const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.9, 2)); // 120BPM 근처 선호
    const score = s * prior;
    if (score > bestScore) { bestScore = score; best = bpm; }
  }
  return Math.round(best * 10) / 10;
}

/* ---------------- 전체 실행 ---------------- */
/**
 * @param {File} file 오디오 파일
 * @param {object} opts { startSec, endSec, stems: ['vocals',...], bpm (0=자동), device: 'auto'|'wasm', modelUrl, signal }
 * @param {object} ui { stage(id, text), progress(id, frac, detail), log(text), stem(name, stereo) }
 */
export async function run(file, opts, ui) {
  const signal = opts.signal;
  ui.stage('decode', '오디오 읽는 중');
  const decoded = await decodeFile(file);
  checkAbort(signal);
  const start = Math.max(0, Math.min(decoded.duration, opts.startSec || 0));
  const end = opts.endSec && opts.endSec > start ? Math.min(decoded.duration, opts.endSec) : decoded.duration;
  const dur = end - start;
  if (dur < 2) throw new Error('구간이 너무 짧습니다. 2초 이상으로 지정해 주세요.');
  const stereo = await render(decoded, SR, 2, start, dur);
  ui.progress('decode', 1, Math.round(dur) + '초 · 44.1kHz 스테레오');

  const { stems: sepr, mode } = await separate(stereo, opts, ui);
  checkAbort(signal);
  const order = mode === '6' ? STEMS6 : STEMS;

  let bpm = opts.bpm;
  if (!bpm) {
    const mono = await render(stereo, BP_SR, 1, 0, stereo.duration);
    bpm = estimateTempo(mono.getChannelData(0), BP_SR);
  }
  ui.log('템포: ' + bpm + ' BPM' + (opts.bpm ? ' (직접 입력)' : ' (자동 추정)'));

  const secToTick = s => Math.max(0, Math.round(s * bpm / 60 * PPQ));
  const tracks = [];
  const wanted = (opts.stems && opts.stems.length ? opts.stems : order).slice();
  // 4트랙으로 돌아왔는데 기타·피아노를 원했다면, 둘이 들어 있는 other를 대신 변환
  if (mode === '4' && (wanted.includes('guitar') || wanted.includes('piano')) && !wanted.includes('other')) wanted.push('other');

  for (const name of order) {
    const st = sepr[name];
    if (!st) continue;
    ui.stem(name, st);
    if (!wanted.includes(name)) continue;
    const id = 'midi-' + name;
    if (rms(st) < SILENCE_RMS) {
      ui.stage(id, STEM_LABELS[name] + ': 거의 무음이라 건너뜀');
      ui.progress(id, 1, '건너뜀');
      continue;
    }
    ui.stage(id, STEM_LABELS[name] + ' 음표 변환 중');
    let notes;
    if (name === 'drums') {
      notes = await transcribeDrums(st, ui);
      ui.progress(id, 1, notes.length + '개 음표');
    } else {
      notes = await transcribePitched(st, STEM_CONFIG[name], opts.modelUrl, p => ui.progress(id, p, Math.round(p * 100) + '%'), signal);
      ui.progress(id, 1, notes.length + '개 음표');
    }
    checkAbort(signal);
    if (!notes.length) continue;
    const isDrum = name === 'drums';
    const cfg = Object.assign({}, STEM_CONFIG[name] || { name: 'Drums', program: 0 });
    if (name === 'other' && mode === '6') cfg.program = 48; // 6트랙에서 other는 주로 스트링·신스
    tracks.push({
      key: name,
      name: isDrum ? 'Drums' : cfg.name,
      channel: isDrum ? 9 : tracks.length >= 9 ? tracks.length + 1 : tracks.length,
      program: cfg.program,
      isDrum,
      notes: notes.map(n => {
        const s = secToTick(n.start);
        return { pitch: n.pitch, start: s, dur: Math.max(1, secToTick(n.end) - s), vel: n.vel };
      }),
    });
    await nextFrame();
  }

  const song = { ppq: PPQ, tempos: [{ tick: 0, uspq: Math.round(60e6 / bpm) }], ts: [4, 4], tracks };
  return { song, bpm, duration: dur, stems: sepr, mode };
}

export const internals = { makeMonophonic, pickPeaks, onsetEnvelope, estimateTempo, encodeWav };
export { tf };
