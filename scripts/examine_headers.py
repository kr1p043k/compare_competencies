from pypdf import PdfReader
import sys, os, json, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

f = "temp/РПД_Базы данных и СУБД.pdf"
if not os.path.exists(f):
    pk = urllib.parse.quote("https://disk.360.yandex.ru/d/-5D0p0XfTwL5Qg", safe="")
    n = urllib.parse.quote("РПД_Базы данных и СУБД.pdf", safe="")
    fp = urllib.parse.quote("/РПД, РПП, ГИА/", safe="")
    url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + pk + "&path=" + fp + n
    data = json.loads(urllib.request.urlopen(url).read())
    urllib.request.urlretrieve(data["href"], f)

reader = PdfReader(f)
print("Total pages:", len(reader.pages))

for i in range(min(12, len(reader.pages))):
    text = reader.pages[i].extract_text()
    lines = text.split("\n")
    print(f"\n=== PAGE {i+1} ===")
    for line in lines:
        s = line.strip()
        if s and (s.isupper() or (len(s) > 3 and s[0].isdigit())):
            print("  " + s[:150])
