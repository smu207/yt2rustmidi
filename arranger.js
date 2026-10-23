/*
 * 솔로 편곡기 — 여러 트랙(합주)을 악기 하나로 칠 수 있는 한 트랙(솔로곡)으로 바꿉니다.
 *
 * 1) 멜로디: '멜로디' 역할 트랙에서 가장 높은 음 한 줄만 뽑습니다.
 * 2) 코드: 반주·베이스 트랙의 음을 구간(1박/2박/1마디)마다 모아 장·단3화음을 찾습니다.
 * 3) 배치: 악기 음역 안에서 멜로디는 위쪽, 베이스는 맨 아래, 코드는 그 사이에 놓습니다.
 *    코드는 앞 코드와 가장 가까운 자리바꿈을 골라 손이 덜 움직이게 합니다.
 * 4) 리듬: 블록 코드 / 쿵짝(베이스-코드) / 아르페지오 / 멜로디만 중에서 고릅니다.
 * 5) 제한: 동시에 울리는 음 수와 음역을 넘지 않게 정리합니다. 멜로디가 항상 우선입니다.
 *
 * DOM을 쓰지 않는 순수 함수라 node에서도 테스트할 수 있습니다.
 */
var Arranger = (function () {
  'use strict';
  const QUALITIES = [
    { name: '', tones: [0, 4, 7] },   // 장3화음
    { name: 'm', tones: [0, 3, 7] },  // 단3화음
  ];
  const PC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  const overlap = (a0, a1, b0, b1) => Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
  const end = n => n.start + n.dur;
  const mod12 = x => ((x % 12) + 12) % 12;

  function guessRole(t) {
    if (t.isDrum) return 'skip';
    const n = (t.name || '').toLowerCase();
    if (/solo|솔로/.test(n)) return 'skip';
    if (/vocal|voice|보컬|melody|멜로디|lead|리드/.test(n)) return 'melody';
    if (/bass|베이스/.test(n)) return 'bass';
    return 'harmony';
  }

  // 음 높이를 [lo, hi] 안으로 옥타브 이동
  function fold(p, lo, hi) {
    while (p > hi) p -= 12;
    while (p < lo) p += 12;
    return p > hi ? p - 12 : p; // 음역이 12반음보다 좁을 때 대비
  }

  // 가장 높은 음 한 줄 (겹치면 높은 음 우선)
  function topVoice(notes, minGap, minDur) {
    const ns = notes.map(n => ({ pitch: n.pitch, start: n.start, dur: n.dur, vel: n.vel }))
      .sort((a, b) => a.start - b.start || b.pitch - a.pitch);
    const out = [];
    for (const n of ns) {
      const prev = out[out.length - 1];
      if (prev && n.start < end(prev)) {
        if (n.start - prev.start < minGap) { if (n.pitch > prev.pitch) out[out.length - 1] = n; continue; }
        if (n.pitch >= prev.pitch) { prev.dur = n.start - prev.start; out.push(n); }
        else if (end(n) > end(prev) + minGap) { const s = end(prev); out.push({ pitch: n.pitch, start: s, dur: end(n) - s, vel: n.vel }); }
      } else out.push(n);
    }
    return out.filter(n => n.dur >= minDur);
  }

  function detectChord(t0, t1, pools) {
    const w = new Array(12).fill(0), wb = new Array(12).fill(0);
    let tot = 0;
    pools.forEach(({ notes, weight, isBass }) => {
      for (const n of notes) {
        if (n.start >= t1) continue;
        const ov = overlap(t0, t1, n.start, end(n));
        if (!ov) continue;
        const v = ov * (0.4 + 0.6 * n.vel / 127) * weight;
        w[mod12(n.pitch)] += v; tot += v;
        if (isBass) wb[mod12(n.pitch)] += v;
      }
    });
    if (tot < 0.12 * (t1 - t0)) return null;
    let best = null;
    for (let r = 0; r < 12; r++) for (const q of QUALITIES) {
      const tones = q.tones.map(x => (r + x) % 12);
      const inside = w[tones[0]] * 1.0 + w[tones[1]] * 0.9 + w[tones[2]] * 0.75;
      const outside = w.reduce((s, v, i) => tones.includes(i) ? s : s + v, 0);
      const score = inside - 0.55 * outside + (wb[r] > 0 ? 0.25 * wb[r] : 0);
      if (!best || score > best.score) best = { root: r, quality: q.name, tones, score };
    }
    if (best.score <= 0) return null;
    let bassPc = best.root, bmax = 0;
    wb.forEach((v, i) => { if (v > bmax) { bmax = v; bassPc = i; } });
    best.bassPc = bmax > 0 ? bassPc : best.root;
    best.label = PC[best.root] + best.quality + (best.bassPc !== best.root ? '/' + PC[best.bassPc] : '');
    return best;
  }

  // 코드 음을 region 안에 배치: 음 사이 간격이 좁고 앞 코드와 가까운 자리를 고름
  function voice(pcs, lo, hi, prevCenter) {
    const options = pcs.map(pc => { const a = []; for (let p = lo; p <= hi; p++) if (mod12(p) === pc) a.push(p); return a; });
    if (options.some(a => !a.length)) return null;
    let best = null;
    const walk = (i, cur) => {
      if (i === options.length) {
        const s = cur.slice().sort((a, b) => a - b);
        if (new Set(s).size !== s.length) return;
        const spread = s[s.length - 1] - s[0];
        if (spread > 12) return;
        const center = s.reduce((x, y) => x + y, 0) / s.length;
        const cost = spread * 0.3 + (prevCenter == null ? Math.abs(center - (lo + hi) / 2) * 0.5 : Math.abs(center - prevCenter));
        if (!best || cost < best.cost) best = { notes: s, cost, center };
        return;
      }
      for (const p of options[i]) { cur.push(p); walk(i + 1, cur); cur.pop(); }
    };
    walk(0, []);
    return best;
  }

  /**
   * @param song { ppq, ts, tempos, tracks:[{name,isDrum,notes:[{pitch,start,dur,vel}]}] }
   * @param o {
   *   roles: ['melody'|'harmony'|'bass'|'skip', ...] (트랙 순서대로, 없으면 이름으로 추측)
   *   lo, hi: 악기 음역 (MIDI 번호), poly: 동시에 낼 수 있는 음 수 (1~6)
   *   style: 'block' | 'beat' | 'arp' | 'none', windowBeats: 1 | 2 | 4, bass: bool
   *   transpose: 'octave' | 'semitone' | 'none', quantize: 0 | 2 | 4 (박당 칸 수), program
   * }
   */
  function arrange(song, o) {
    const ppq = song.ppq, ts = song.ts || [4, 4];
    const beat = ppq * 4 / ts[1], bar = beat * ts[0];
    const lo = Math.max(0, Math.min(o.lo, o.hi)), hi = Math.min(127, Math.max(o.lo, o.hi));
    if (hi - lo < 11) throw new Error('음역이 너무 좁습니다. 최소 한 옥타브(12반음) 이상으로 정해 주세요.');
    const poly = Math.max(1, Math.min(6, o.poly | 0 || 1));
    const roles = song.tracks.map((t, i) => (o.roles && o.roles[i]) || guessRole(t));
    const info = { notes: [] };

    const pick = r => song.tracks.filter((t, i) => roles[i] === r && !t.isDrum).flatMap(t => t.notes);
    let melSrc = pick('melody');
    const har = pick('harmony'), bas = pick('bass');
    if (!melSrc.length) {
      if (!har.length) throw new Error('멜로디로 쓸 트랙이 없습니다. 트랙 역할에서 하나를 "멜로디"로 정해 주세요.');
      melSrc = har;
      info.notes.push('멜로디 트랙이 없어 반주 트랙의 가장 높은 음을 멜로디로 썼습니다.');
    }

    // 1) 멜로디
    const g = o.quantize ? ppq / o.quantize : 0;
    const q = t => g ? Math.round(t / g) * g : t;
    // 아주 짧거나(1/32음표 미만) 아주 약한 음은 잡음으로 보고 멜로디에서 뺌
    let melody = topVoice(melSrc.filter(n => n.vel >= 20), ppq / 16, ppq / 8);
    if (g) {
      melody = melody.map(n => { const s = q(n.start); return { ...n, start: s, dur: Math.max(g, q(end(n)) - s) }; });
      for (let i = 0; i + 1 < melody.length; i++) if (end(melody[i]) > melody[i + 1].start) melody[i].dur = Math.max(1, melody[i + 1].start - melody[i].start);
      melody = melody.filter((n, i) => i === 0 || n.start !== melody[i - 1].start);
    }
    if (!melody.length) throw new Error('멜로디 음표가 없습니다.');

    // 멜로디가 놓일 위쪽 음역
    const accomp = poly > 1 && (o.style !== 'none' || o.bass);
    const melLo = accomp ? lo + Math.min(19, Math.round((hi - lo) * 0.42)) : lo;

    // 2) 조옮김: 멜로디가 위쪽 음역에 가장 많이 들어가는 만큼 이동
    let shift = 0;
    if (o.transpose !== 'none') {
      const step = o.transpose === 'semitone' ? 1 : 12;
      let bestScore = -Infinity;
      const totalDur = melody.reduce((s, n) => s + n.dur, 0);
      for (let sh = -48; sh <= 48; sh += step) {
        let inR = 0;
        for (const n of melody) { const p = n.pitch + sh; if (p >= melLo && p <= hi) inR += n.dur; }
        const score = inR - Math.abs(sh) * totalDur * 0.002 - (sh % 12 ? totalDur * 0.03 : 0);
        if (score > bestScore) { bestScore = score; shift = sh; }
      }
    }
    info.shift = shift;

    let folded = 0;
    const out = melody.map(n => {
      const p0 = n.pitch + shift, p = fold(p0, melLo, hi);
      if (p !== p0) folded++;
      return { pitch: p, start: n.start, dur: n.dur, vel: 100, role: 0 };
    });
    info.folded = folded;
    info.melodyNotes = out.length;

    // 3) 코드
    const songEnd = Math.max(...song.tracks.flatMap(t => t.notes.map(end)), ...out.map(end));
    const win = beat * (o.windowBeats || 2);
    const pools = [
      { notes: har, weight: 1 },
      { notes: bas, weight: 1.4, isBass: true },
      { notes: melSrc === har ? [] : melSrc, weight: 0.35 },
    ];
    const segs = [];
    if (accomp) {
      for (let t = 0; t < songEnd; t += win) {
        const c = detectChord(t, Math.min(songEnd, t + win), pools);
        const last = segs[segs.length - 1];
        if (last && c && last.chord && last.chord.label === c.label && o.style === 'block') last.end = t + win;
        else segs.push({ start: t, end: t + win, chord: c });
      }
    }
    info.chords = segs.filter(s => s.chord).length;
    info.chordLabels = segs.filter(s => s.chord).slice(0, 16).map(s => s.chord.label);

    // 반주 음이 멜로디보다 위로 가지 않게: 그 시점 멜로디 음
    const melAt = t => { for (const n of out) { if (n.start > t) break; if (end(n) > t) return n.pitch; } return null; };

    const acc = [];
    let prevCenter = null;
    const bassOn = !!o.bass && poly >= 2;
    // 코드에 쓸 수 있는 음 수 (쿵짝·아르페지오는 베이스와 코드가 동시에 울리지 않음)
    const budget = o.style === 'beat' || o.style === 'arp' ? poly - 1 : poly - 1 - (bassOn ? 1 : 0);
    const bassLo = lo, bassHi = lo + 11;
    const cLo = Math.max(lo, bassOn ? Math.min(lo + 10, melLo - 7) : lo);
    const cHi = Math.max(cLo + 11, melLo + 2);
    for (const seg of segs) {
      const c = seg.chord;
      if (!c) continue;
      const bassP = bassOn ? fold(c.bassPc, bassLo, bassHi) : null;
      let chordPitches = [];
      if (o.style !== 'none' && budget > 0) {
        let pcs = c.tones.slice();
        if (o.style === 'arp') pcs = c.tones.slice();
        else if (budget === 1) pcs = [c.tones[1]];
        else if (budget === 2) pcs = bassOn ? [c.tones[1], c.tones[2]] : [c.tones[0], c.tones[1]];
        const v = voice(pcs, cLo, cHi, prevCenter);
        if (v) { chordPitches = v.notes; prevCenter = v.center; }
      }
      const add = (pitch, start, dur, vel, role) => {
        if (start >= seg.end || dur <= 0) return;
        const m = melAt(start);
        if (m != null && pitch >= m) { pitch -= 12; if (pitch < lo) return; }
        if (bassP != null && role === 2 && pitch <= bassP) return;
        acc.push({ pitch, start, dur: Math.min(dur, seg.end - start), vel, role });
      };
      const len = seg.end - seg.start;
      if (o.style === 'arp') {
        const seq = [bassP, ...chordPitches, ...chordPitches.slice(0, -1).reverse().slice(0, -1)].filter(p => p != null);
        const step = beat / 2;
        let k = 0;
        for (let t = seg.start; t < seg.end - 1; t += step, k++) add(seq[k % seq.length], t, step * 0.95, k % seq.length === 0 && bassP != null ? 82 : 64, k % seq.length === 0 && bassP != null ? 1 : 2);
      } else if (o.style === 'beat') {
        for (let t = seg.start, i = Math.round((seg.start % bar) / beat); t < seg.end - 1; t += beat, i++) {
          const strong = i % 2 === 0;
          if (bassP != null && strong) add(bassP, t, beat * 0.9, 84, 1);
          if (!strong || bassP == null) chordPitches.forEach(p => add(p, t, beat * 0.7, 62, 2));
        }
      } else { // block 또는 none(베이스만)
        if (bassP != null) add(bassP, seg.start, len - ppq / 16, 82, 1);
        if (o.style === 'block') chordPitches.forEach(p => add(p, seg.start, len - ppq / 16, 60, 2));
      }
    }

    // 4) 동시음 제한: 멜로디 > 베이스 > 코드, 코드는 낮은 음부터 뺌
    let all = out.concat(acc).sort((a, b) => a.start - b.start || a.role - b.role || b.pitch - a.pitch);
    const kept = [];
    for (const n of all) {
      const active = kept.filter(k => k.start <= n.start && end(k) > n.start);
      if (active.some(k => k.pitch === n.pitch)) {
        if (n.role === 0) { active.filter(k => k.pitch === n.pitch).forEach(k => { k.dur = Math.max(1, n.start - k.start); }); }
        else continue;
      }
      const sounding = active.filter(k => end(k) > n.start && k.pitch !== n.pitch);
      if (sounding.length + 1 > poly) {
        if (n.role === 0) {
          // 멜로디가 들어갈 자리를 만들기 위해 가장 덜 중요한 음을 줄임
          const victim = sounding.filter(k => k.role > 0).sort((a, b) => b.role - a.role || a.pitch - b.pitch)[0];
          if (victim) victim.dur = Math.max(1, n.start - victim.start);
        } else continue;
      }
      kept.push(n);
    }
    const notes = kept.filter(n => n.dur > 0 && n.pitch >= lo && n.pitch <= hi)
      .map(n => ({ pitch: n.pitch, start: Math.round(n.start), dur: Math.max(1, Math.round(n.dur)), vel: n.vel }))
      .sort((a, b) => a.start - b.start || a.pitch - b.pitch);

    info.total = notes.length;
    info.maxPoly = maxPolyphony(notes);
    return {
      track: { name: o.name || 'Solo (편곡)', channel: 0, program: o.program == null ? 0 : o.program, isDrum: false, notes },
      info,
    };
  }

  function maxPolyphony(notes) {
    const ev = [];
    notes.forEach(n => { ev.push([n.start, 1]); ev.push([end(n), -1]); });
    ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let c = 0, m = 0;
    ev.forEach(e => { c += e[1]; if (c > m) m = c; });
    return m;
  }

  // 게임 안에서 실제로 소리 나는 음역을 확인하는 반음계 MIDI (C1~C8, 한 음에 0.5초)
  function rangeTestSong(lo, hi) {
    lo = lo == null ? 24 : lo; hi = hi == null ? 108 : hi;
    const ppq = 480, notes = [];
    let t = 0;
    for (let p = lo; p <= hi; p++) {
      const isC = p % 12 === 0;
      notes.push({ pitch: p, start: t, dur: isC ? ppq * 1.8 : ppq * 0.9, vel: isC ? 115 : 85 });
      t += isC ? ppq * 2 : ppq; // C는 길게 쳐서 옥타브를 세기 쉽게
    }
    return { ppq, tempos: [{ tick: 0, uspq: 500000 }], ts: [4, 4], tracks: [{ name: 'Range test C' + (Math.floor(lo / 12) - 1) + '-C' + (Math.floor(hi / 12) - 1), channel: 0, program: 0, isDrum: false, notes }] };
  }

  return { arrange, guessRole, rangeTestSong, maxPolyphony, _internal: { topVoice, detectChord, voice, fold } };
})();
if (typeof module !== 'undefined') module.exports = Arranger;
