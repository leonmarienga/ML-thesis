import requests
from bs4 import BeautifulSoup
for url in ["https://www.nhc.noaa.gov/archive/2022.shtml","https://www.nhc.noaa.gov/archive/2022/","https://www.nhc.noaa.gov/archive/2022/IAN.shtml"]:
 try:
  r=requests.get(url,timeout=20);print("URL",url,"STATUS",r.status_code,"LENGTH",len(r.content),flush=True)
  s=BeautifulSoup(r.text,"html.parser")
  print("TITLE",s.title.get_text(" ",strip=True) if s.title else "NONE")
  for a in s.find_all("a",href=True):
   t=a.get_text(" ",strip=True)
   if "IAN" in t.upper() or "PUBLIC" in t.upper() or "0900 UTC" in t:
    print("ANCHOR",repr(t),repr(a.get("href")),str(a.parent)[:300].replace("\n"," "))
    if "PUBLIC" in t.upper():print("PARENT_PARENT",str(a.parent.parent)[:800].replace("\n"," "))
  if "IAN.shtml" in url:
   print("INDEX_HTML",r.text[r.text.find("Friday, September 23")-400:r.text.find("Friday, September 23")+2400][:2800].replace("\n"," "))
 except Exception as e:print("ERR",str(e)[:200],flush=True)
