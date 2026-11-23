import requests, re, json

url = 'https://data.mendeley.com/datasets/y58jknpgs8/2/files'
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(url, headers=headers, timeout=15)
html = r.text

# Try to find download links
xlsx_links = re.findall(r'"downloadUrl":"(.*?)"', html)
print("Download URLs found:", xlsx_links[:5])

file_ids = re.findall(r'"fileId":"([^"]+)"', html)
print("File IDs:", file_ids[:5])

# Look for the data embedded in Next.js __NEXT_DATA__
next_data = re.findall(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
if next_data:
    try:
        data = json.loads(next_data[0])
        print("Found __NEXT_DATA__")
        # Navigate to find files
        props = data.get('props', {}).get('pageProps', {})
        dataset = props.get('dataset', {})
        files = dataset.get('files', [])
        print(f"Number of files: {len(files)}")
        for f in files[:10]:
            print(f" - {f.get('filename','?')} | size={f.get('fileSize','?')} | id={f.get('id','?')}")
    except Exception as e:
        print(f"JSON parse error: {e}")
else:
    print("No __NEXT_DATA__ found")
    # Try alternate approach - look for file names
    filenames = re.findall(r'"filename":"([^"]+)"', html)
    print("Filenames found:", filenames[:10])
