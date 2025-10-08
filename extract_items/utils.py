import base64


def get_image_data_uri(uploaded_file):

    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_file.content_type
    return f"data:{mime_type};base64,{encoded_image}"