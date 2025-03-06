from capybaradb._emb_json._emb_image import EmbImage
import base64

# Test valid mime type
try:
    img = EmbImage('SGVsbG8gV29ybGQ=', mime_type='image/png')
    print(f'Created EmbImage with mime_type: {img.mime_type}')
    print(img.to_json())
except ValueError as e:
    print(f'Unexpected error: {e}')

# Test invalid mime type
try:
    img = EmbImage('SGVsbG8gV29ybGQ=', mime_type='image/invalid')
    print('This should not print')
except ValueError as e:
    print(f'Validation error (expected): {e}')

# Test from_json
try:
    json_dict = {
        "data": "SGVsbG8gV29ybGQ=",
        "mime_type": "image/jpeg",
        "chunks": ["test chunk"],
    }
    img = EmbImage.from_json(json_dict)
    print(f'Created EmbImage from JSON with mime_type: {img.mime_type}')
    print(f'Chunks: {img.chunks}')
except Exception as e:
    print(f'Error in from_json: {e}') 