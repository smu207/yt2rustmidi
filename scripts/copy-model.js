// Basic Pitch 모델(약 0.9MB)을 node_modules 에서 model/basic-pitch 로 복사합니다.
const fs = require('fs');
const path = require('path');
const src = path.join(__dirname, '..', 'node_modules', '@spotify', 'basic-pitch', 'model');
const dst = path.join(__dirname, '..', 'model', 'basic-pitch');
fs.mkdirSync(dst, { recursive: true });
for (const f of fs.readdirSync(src)) fs.copyFileSync(path.join(src, f), path.join(dst, f));
fs.copyFileSync(path.join(src, '..', 'LICENSE'), path.join(dst, 'LICENSE'));
console.log('Basic Pitch 모델 복사 완료 →', dst);
