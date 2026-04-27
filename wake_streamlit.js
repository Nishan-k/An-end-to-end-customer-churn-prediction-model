const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log(`Opening ${process.env.APP_URL}...`);
  await page.goto(process.env.APP_URL, { waitUntil: 'networkidle' });

  
  const wakeButton = page.locator('button:has-text("Yes, get this app back up")');
  if (await wakeButton.count() > 0) {
    console.log('Wake-up button found – clicking it.');
    await wakeButton.click();
    await page.waitForTimeout(5000); 
  } else {
    console.log('App already awake – no button needed.');
  }

  await browser.close();
})();