const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const outDir = path.resolve(__dirname, 'screenshots');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

async function run() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    args: ['--no-sandbox', '--disable-gpu', '--window-size=1440,960'],
    defaultViewport: { width: 1440, height: 960 }
  });

  const page = await browser.newPage();

  // Load first to get run IDs from backend
  await page.goto('http://127.0.0.1:5174/#runs', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 1200));

  // Extract a run ID from the page or storage
  const runId = await page.evaluate(async () => {
    try {
      const res = await fetch('/api/runs?limit=1');
      const data = await res.json();
      return (data.runs && data.runs[0] && data.runs[0].run_id) || null;
    } catch (e) {
      return null;
    }
  });
  console.log('Sample run ID:', runId);

  // Set theme to dark in localStorage
  await page.evaluate(() => {
    localStorage.setItem('blackbox_theme', 'dark');
    document.documentElement.setAttribute('data-theme', 'dark');
  });

  const views = [
    { name: '01_home_dark', url: 'http://127.0.0.1:5174/#home' },
    { name: '02_runs_overview_dark', url: 'http://127.0.0.1:5174/#runs' },
    { name: '03_run_detail_dark', url: runId ? `http://127.0.0.1:5174/#run/${runId}` : 'http://127.0.0.1:5174/#runs' },
    { name: '04_live_execution_dark', url: 'http://127.0.0.1:5174/#live' },
    { name: '05_replay_alternatives_dark', url: 'http://127.0.0.1:5174/#replay' },
    { name: '06_trace_comparison_dark', url: 'http://127.0.0.1:5174/#compare' },
    { name: '07_failure_lab_dark', url: 'http://127.0.0.1:5174/#break' },
    { name: '08_model_evaluation_dark', url: 'http://127.0.0.1:5174/#evaluation' },
  ];

  for (const v of views) {
    console.log(`Capturing ${v.name}...`);
    await page.goto(v.url, { waitUntil: 'networkidle0' });
    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'dark');
    });
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(outDir, `${v.name}.png`), fullPage: false });
  }

  // Also capture light mode for comparison on Home and Runs
  await page.evaluate(() => {
    localStorage.setItem('blackbox_theme', 'light');
    document.documentElement.setAttribute('data-theme', 'light');
  });
  await page.goto('http://127.0.0.1:5174/#runs', { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(outDir, '00_runs_overview_light.png') });

  await browser.close();
  console.log('All screenshots captured successfully in', outDir);
}

run().catch(err => {
  console.error('Error taking screenshots:', err);
  process.exit(1);
});
