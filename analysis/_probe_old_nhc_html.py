import requests,re
from bs4 import BeautifulSoup
cases=[(2012,"SANDY"),(2017,"IRMA"),(2018,"FLORENCE"),(2020,"LAURA"),(2021,"IDA"),(2022,"IAN")]
for y,n in cases:
 u=f"https://www.nhc.noaa.gov/archive/{y}/{n}.shtml"
 try:
  r=requests.get(u,timeout=20)
  s=r.text
  print("STORM",y,n,"status",r.status_code,"len",len(s),"public_occurrences",s.lower().count(".public."),"COMMENTS",len(re.findall(r"<!--",s)),"UTC",s.count("UTC"))
  for pattern in [r".{0,175}\.public.{0,155}",r".{0,140}public_advisory.{0,200}",r".{0,140}Public Advisory.{0,200}"]:
   found=re.findall(pattern,s,re.I|re.S)
   for x in found[:2]:print("MATCH",re.sub(r"\s+"," ",x)[:400])
  links=[(x.get("href"),x.get_text(" ",strip=True)) for x in BeautifulSoup(s,"html.parser").find_all("a",href=True) if "public" in x.get("href","").lower()]
  print("PUBLIC LINKS",len(links),links[:2])
 except Exception as e:print("ERROR",repr(e))
