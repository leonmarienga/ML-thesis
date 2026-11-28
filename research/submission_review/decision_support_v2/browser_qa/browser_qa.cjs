const fs=require('node:fs');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const {resolve}=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('playwright');
async function main(){
 const browserPkg=(await import('@sparticuz/chromium')).default;
 const html=resolve(process.argv[2]??'index.html');
 const hash=crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex');
 assert.equal(hash,'9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203');
 const browser=await chromium.launch({executablePath:await browserPkg.executablePath(),args:browserPkg.args,headless:true});
 const checks=[];
 try {
  for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
   const page=await browser.newPage({viewport});
   const errors=[], requests=[];
   page.on('pageerror',e=>errors.push(e.message));
   page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
   await page.route(/^https?:/,r=>r.abort());
   await page.goto(pathToFileURL(html).href,{waitUntil:'load'});
   assert.match(await page.locator('h1').innerText(),/Disaster relief decision support/);
   assert.match(await page.locator('#interval-rows').innerText(),/281\s*\/\s*308/);
   assert.match(await page.locator('#interval-rows').innerText(),/0\s*\/\s*9/);
   let focused='';
   for(let i=0;i<24;i++){
    await page.keyboard.press('Tab');
    focused=await page.evaluate(()=>document.activeElement?.id??'');
    if(focused==='interval-level')break;
   }
   assert.equal(focused,'interval-level','interval switch reachable from keyboard');
   await page.keyboard.press('ArrowDown');
   assert.equal(await page.locator('#interval-level').inputValue(),'80');
   assert.match(await page.locator('#interval-message').innerText(),/80%/);
   await page.keyboard.press('ArrowUp');
   assert.equal(await page.locator('#interval-level').inputValue(),'90');
   await page.locator('#veto-regime').focus();
   await page.keyboard.press('ArrowDown');
   assert.notEqual(await page.locator('#veto-regime').inputValue(),'all');
   const subset=await page.locator('#veto-rows').innerText();
   const subsetCaption=await page.locator('#veto-caption').innerText();
   assert.doesNotMatch(subsetCaption,/48 matched/);
   assert.doesNotMatch(subset,/\[[^\]]+\]/);
   await page.keyboard.press('Home'); 
   assert.equal(await page.locator('#veto-regime').inputValue(),'all');
   assert.match(await page.locator('#veto-caption').innerText(),/48 matched/);
   await page.locator('#comparator').focus();
   const previous=await page.locator('#comparator').inputValue();
   await page.keyboard.press('ArrowDown');
   assert.notEqual(await page.locator('#comparator').inputValue(),previous);
   const dims=await page.evaluate(()=>({viewport:innerWidth,body:document.documentElement.scrollWidth}));
   assert.equal(errors.length,0,JSON.stringify(errors));
   assert.equal(requests.length,0,JSON.stringify(requests));
   await page.screenshot({path:'qa-'+viewport.width+'.png',fullPage:true});
   checks.push({viewport,dims,keyboardInterval:true,keyboardRoadFilter:true,keyboardComparator:true,pageErrors:errors,httpRequests:requests});
   await page.close();
  }
  console.log(JSON.stringify({engine:browser.version(),hash,checks},null,2));
 } finally { await browser.close(); }
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
