with open("data/reference/krm_disciplines_09.03.02.json", "rb") as f:
    raw = f.read()

idx = raw.find(b"09.03.02")
print("Bytes at 09.03.02:")
print(raw[idx:idx+80])
print("Hex:", raw[idx:idx+80].hex())

# Check if skills contain proper UTF-8 Russian
idx2 = raw.find("современные".encode("utf-8"))
print("\nFound 'современные' at:", idx2)
if idx2 >= 0:
    print("Context:", raw[max(0,idx2-10):idx2+60])
else:
    # Check for common Russian words
    for word in ["современ", "технологи", "данных", "SQL"]:
        e = word.encode("utf-8")
        i = raw.find(e)
        print(f"  '{word}' found at: {i}")
    
    # Check encoding by looking at high bytes
    count_high = sum(1 for b in raw if b > 127)
    print(f"\nTotal high bytes: {count_high} / {len(raw)}")
    
    # Check first few high-byte patterns
    for i, b in enumerate(raw):
        if b > 127:
            print(f"First high byte at {i}: {raw[max(0,i-5):i+15]}")
            print(f"  hex: {raw[max(0,i-5):i+15].hex()}")
            break
