import base64


def to_data_uri(uploaded_file, mime_type="image/jpeg"):
    uploaded_file.seek(0)
    b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"