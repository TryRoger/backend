import glob
import time
from google import genai
from google.genai import types

client = genai.Client()

start_time = time.time()

# Find all matching images
image_paths = sorted(glob.glob("/Users/user/tryroger/mouse/mac_app/swift_app/backend/images/f280a9dc*"))

# Build contents with all images as inline data
contents = []
count = 0
for image_path in image_paths:
    count = count + 1
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    contents.append(
        types.Part.from_bytes(
            data=img_bytes,
            mime_type='image/png'
        )
    )

print(f"total_images {count}")
contents.append("What's happening?")

# Send to Gemini
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=contents
)

end_time = time.time()
print(response.text)
print(f"\nTime taken: {end_time - start_time:.2f}s")
