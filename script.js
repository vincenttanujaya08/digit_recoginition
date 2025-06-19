// script.js

// 1. Ambil elemen DOM
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
const resetBtn = document.getElementById('reset-btn');
const analisaBtn = document.getElementById('analisa-btn');
const digitResult = document.getElementById('digit-result');

// 2. Inisialisasi kanvas → blank putih
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

let drawing = false;

// 3. Event menggambar di kanvas
canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
});

canvas.addEventListener('mouseleave', () => {
  drawing = false;
});

// 4. Reset kanvas
resetBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  digitResult.innerText = '—';
});

// 5. URL backend Flask (sesuaikan host/port jika berbeda)
const FLASK_URL = 'http://127.0.0.1:5001/predict';

// 6. Saat tombol “Analisa” diklik:
analisaBtn.addEventListener('click', async () => {
  // a. Buat canvas offscreen 28×28
  const off = document.createElement('canvas');
  off.width = 28;
  off.height = 28;
  const offCtx = off.getContext('2d');

  // b. Resize canvas utama (280×280) → 28×28
  offCtx.drawImage(canvas, 0, 0, 28, 28);

  // c. Ambil data pixel dan bentuk array 2D [28][28]
  const imgData = offCtx.getImageData(0, 0, 28, 28).data;
  const arr2d = [];
  for (let y = 0; y < 28; y++) {
    const row = [];
    for (let x = 0; x < 28; x++) {
      const idx = (y * 28 + x) * 4;
      // Ambil channel R → invert: hitam→1, putih→0
      row.push((255 - imgData[idx]) / 255);
    }
    arr2d.push(row);
  }

  // Informasi menunggu
  digitResult.innerText = 'Menebak…';

  try {
    // d. Kirim request POST
    const res = await fetch(FLASK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: arr2d })
    });

    if (!res.ok) {
      // Jika server merespons error HTTP (400/500), tampilkan kode status
      const errJson = await res.json().catch(() => null);
      console.error('Server respon status', res.status, errJson);
      digitResult.innerText = `Error ${res.status}`;
      return;
    }

    const data = await res.json();
    console.log('Response server:', data);

    // e. Jika backend mengembalikan digit=null, minta gambar ulang
    if (data.digit === null) {
      digitResult.innerText =
        `Kurang yakin (${(data.confidence * 100).toFixed(1)}%). Silakan gambar ulang.`;
    } else {
      digitResult.innerText =
        `Tebakan: ${data.digit}  —  Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    }
  } catch (err) {
    // Jika fetch gagal (misal server mati)
    console.error('Fetch error:', err);
    digitResult.innerText = 'Gagal koneksi ke server';
  }
});
