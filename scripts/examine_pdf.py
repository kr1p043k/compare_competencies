import json, urllib.request, urllib.parse, sys, re
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pub_key = "https://disk.360.yandex.ru/d/-5D0p0XfTwL5Qg"
enc_key = urllib.parse.quote(pub_key, safe="")
folder = "/РПД, РПП, ГИА/"
enc_folder = urllib.parse.quote(folder, safe="")

names = [
    "РПД_Базы данных и СУБД.pdf",
    "РПД_Алгоритмизация и программирование.pdf",
]

for name in names:
    enc_name = urllib.parse.quote(name, safe="")
    dl_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + enc_key + "&path=" + enc_folder + enc_name
    resp = json.loads(urllib.request.urlopen(dl_url).read())
    urllib.request.urlretrieve(resp["href"], "temp/" + name)

    reader = PdfReader("temp/" + name)
    full_text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text += t + "\n"

    # Find competency section
    sections = re.split(r"\n(?=\d+\.\s*(?:ПЕРЕЧЕНЬ|КОМПЕТЕНЦИИ|ПЛАНИРУЕМЫЕ|РЕЗУЛЬТАТЫ|ЗНАТЬ|УМЕТЬ|ВЛАДЕТЬ|СТРУКТУРА|СОДЕРЖАНИЕ|УЧЕБНО|ОЦЕНОЧНЫЕ|МЕТОДИЧЕСКИЕ))", full_text, flags=re.IGNORECASE)

    print("\n" + "="*60)
    print("FILE: " + name + " (" + str(len(reader.pages)) + " pages)")
    print("="*60)

    for i, sec in enumerate(sections):
        upper = sec[:200].upper()
        if any(kw in upper for kw in ["КОМПЕТЕНЦ", "ПЛАНИРУЕМЫЕ РЕЗУЛЬТАТ", "ИНДИКАТОР"]):
            print("\n--- Section " + str(i) + " ---")
            print(sec[:4000])
