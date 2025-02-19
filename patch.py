
with open(".\\x64\\Debug\\XOR_CUDA_Unpacking_C.exe", "rb") as f:
    data = f.read()
    offset = data.find(b"ST_ART_HE_R_E")
    print(f"Shellcode offset: {hex(offset)}")
with open(".\\x64\\Debug\\XOR_CUDA_Unpacking_C.exe", "rb+") as f:
    f.seek(offset)
    payload = open("test.vdi", "rb").read()
    f.write(payload[:10000000])  # Ensure it doesn't overflow the buffer
