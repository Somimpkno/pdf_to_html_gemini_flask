import fitz  # PyMuPDF
import os
import time
import google.generativeai as genai
import json
from PIL import Image

# --- Gemini API Setup ---
def get_gemini_client_and_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    vision_model = None
    text_model = None
    try:
        # Note: If 'gemini-1.5-flash-latest' is the recommended one, use it.
        # The original script had 'gemini-2.0-flash', which might not be widely available.
        # Using 'gemini-1.5-flash-latest' for vision as a common accessible model.
        vision_model_name = "gemini-2.0-flash" # More likely to be available
        vision_model = genai.GenerativeModel(model_name=vision_model_name)
        print(f"Using Vision Model: {vision_model.model_name}")
    except Exception as e_vis:
        print(f"Could not initialize vision model '{vision_model_name}'. Vision features might be limited. Error: {e_vis}")

    try:
        text_model_name = "gemini-2.5-flash" # Using a more robust model for generation.
                                                  # 'gemini-2.5-flash-preview-05-20' is specific.
                                                  # 'gemini-1.5-pro-latest' is a good general choice.
        text_model = genai.GenerativeModel(model_name=text_model_name)
        print(f"Using Text Model: {text_model.model_name}")
    except Exception as e_text:
        print(f"Could not initialize text model '{text_model_name}'. HTML generation will fail. Error: {e_text}")
        raise

    return vision_model, text_model

# --- Image Extraction and Alt Tag Generation ---
def generate_alt_text_for_local_image(vision_model, image_path, primary_target_language):
    if not vision_model:
        return "Image" # Default if no vision model
    
    # Fallback if primary language is not set, though UI should enforce selection
    primary_lang_for_alt = primary_target_language if primary_target_language else "English"

    # Rate limiting - Gemini API has quotas. 1-2 seconds is often safer than 0.
    # The old script had 4s, which might be too long for a web request.
    # Consider making this configurable or handling API rate limit errors more gracefully.
    time.sleep(4) # Reduced for web context, monitor for rate limit errors.
    print(f"Generating alt text for local image: {os.path.basename(image_path)} in {primary_lang_for_alt}...")
    try:
        img_pil = Image.open(image_path)
        prompt_for_alt_text = f"Provide a single, concise, and descriptive alt text for this image in {primary_lang_for_alt}, suitable for an HTML img tag's 'alt' attribute. Output only the alt text string itself, with no additional explanations, quotation marks, or markdown formatting. Be factual and brief."

        response = vision_model.generate_content([
            prompt_for_alt_text,
            img_pil
        ],
        generation_config={"temperature": 0.3})

        alt_text = response.text.strip().replace('"', '').replace("'", "").replace('\n', ' ').strip()
        print(f"  Alt text for {os.path.basename(image_path)}: '{alt_text}'")
        return alt_text
    except Exception as e:
        print(f"Error generating alt text for {image_path}: {e}")
        return f"Image placeholder - error generating alt text"

def extract_images_and_generate_alt_tags(pdf_path, output_images_folder, vision_model, primary_target_language, html_image_subfolder="extracted_images"):
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}.")
        return []

    os.makedirs(output_images_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    images_data = []
    image_extraction_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        page_image_index = 0
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if not base_image or not base_image.get("image"):
                print(f"Could not extract image data for xref {xref} on page {page_num + 1}")
                continue

            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            image_extraction_counter += 1
            page_image_index += 1

            image_filename = f"page_{page_num + 1}_idx_{page_image_index}_gidx_{image_extraction_counter}.{image_ext}"
            local_image_path = os.path.join(output_images_folder, image_filename)

            with open(local_image_path, "wb") as img_file:
                img_file.write(image_bytes)

            alt_text = generate_alt_text_for_local_image(vision_model, local_image_path, primary_target_language)
            html_relative_path = os.path.join(html_image_subfolder, image_filename) # Path relative to the final HTML

            images_data.append({
                "pdf_page_num": page_num + 1,
                "image_index_on_page": page_image_index,
                "html_src_path": html_relative_path,
                "alt_text": alt_text,
            })
            print(f"Extracted: {local_image_path} (Page: {page_num+1}, Index on page: {page_image_index}), Alt: '{alt_text}'")
    doc.close()
    return images_data

# --- HTML Generation ---
def generate_html_from_pdf_gemini(text_model, uploaded_pdf_file_object, images_metadata_list, target_languages):
    print(f"Generating HTML from PDF: {uploaded_pdf_file_object.name}")
    print(f"Target languages for output: {', '.join(target_languages)}")
    print(f"Using {len(images_metadata_list)} pre-extracted images with alt text.")

    images_metadata_json = json.dumps(images_metadata_list, indent=2, ensure_ascii=False)

    if not target_languages:
        target_languages = ["English"] # Default

    language_instructions = ""
    if len(target_languages) == 1:
        language_instructions = f"The entire HTML content, including all text, should be in {target_languages[0]}."
    else:
        langs_str_list = [f"{lang}" for lang in target_languages]
        langs_str = ", ".join(langs_str_list[:-1]) + " and " + langs_str_list[-1] if len(langs_str_list) > 1 else langs_str_list[0]
        translation_order = "\n".join([f"Then, present its translation in {target_languages[i]}." for i in range(1, len(target_languages))])
        language_instructions = f"""The entire HTML content should be presented in all specified languages: {langs_str}.
For each piece of text content (e.g., paragraph, list item, heading), first present it in {target_languages[0]}.
{translation_order}
Example for a paragraph with English and Hindi:
<p lang="{target_languages[0].lower()[:2]}">This is the English text.</p>
<p lang="{target_languages[1].lower()[:2]}">यह हिंदी में पाठ है।</p>
Maintain this point-by-point or segment-by-segment multilingual presentation throughout the document. Use 'lang' attributes on text elements.
"""

    # Basic head content, can be enhanced in app.py or template
    head_content = f"""<meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Content ({', '.join(target_languages)})</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">
    <script>
        window.MathJax = {{
          tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
          }},
          chtml: {{ matchFontHeight: false, mtextInheritFont: true }},
          svg: {{ mtextInheritFont: true }}
        }};
    </script>
    <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
      body {{ font-family: 'Noto Sans', 'Noto Sans Devanagari', sans-serif; margin: 20px; line-height: 1.6; }}
      .scrollable-table-wrapper {{ overflow-x: auto; margin-bottom: 1em; border: 1px solid #ddd; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
      th {{ background-color: #f2f2f2; }}
      img {{ max-width: 100%; height: auto; display: block; margin: 1em auto; border: 1px solid #eee;}}
      h1, h2, h3, h4, h5, h6 {{ margin-top: 1.5em; margin-bottom: 0.5em; }}
      p {{ margin-bottom: 1em; }}
      ul, ol {{ margin-bottom: 1em; padding-left: 40px; }}
      li {{ margin-bottom: 0.5em; }}
    </style>"""

    system_instruction_text = f"""You are an expert PDF to HTML converter.
Your primary goal is to convert the provided PDF content (given as a file input) and associated image information (given as structured text/JSON) into a single, well-structured, and valid HTML file.
The HTML should accurately replicate the text content, general layout, tables, lists, and headings from the PDF.
ONLY use content and information present in the original PDF. DO NOT add any new data, opinions, or external information.

**Language Output:**
{language_instructions}

**HTML Formatting Rules:**
1.  Use pure HTML tags ONLY. No Markdown syntax.
2.  Structure: Complete HTML document (`<!DOCTYPE html>`, `<html>`, `<head>`, `<body>`). The `<head>` section MUST include:
{head_content}
3.  Equations: Use MathJax compatible LaTeX. Inline: `\\( ... \\)` or `$ ... $`. Display: `\\[ ... \\]` or `$$ ... $$`.
    Example: `<p style="text-align:center;">$\\text{{लवण सूचकांक}} = (\\text{{कुल Na}} - 24.5) - [(\\text{{कुल Ca}} - \\text{{Ca in }} CaCO_3) \\times 4.85]$</p>`
4.  Tables: Convert PDF tables into HTML `<table>`. Wrap wide tables in `<div class="scrollable-table-wrapper">...</div>`.
5.  Text Preservation: Preserve all text exactly as it appears in the PDF, then translate/present according to language instructions. For multi-language, use `lang` attribute on paragraph or span tags for each language segment.
6.  Diagrams (Non-Image): If the PDF contains diagrams made from text, lines, or shapes, attempt to replicate their structure using semantic HTML and CSS. If too complex, describe it briefly in text.

**Image Handling - CRITICAL:**
You have been provided with a JSON list of pre-extracted image metadata.
When you identify an image's position in the PDF content:
- You MUST use the provided metadata to insert an `<img>` tag: `<img src="[html_src_path]" alt="[alt_text]" style="max-width:100%; height:auto; display:block; margin:1em auto;">`
- Match image from PDF context to `pdf_page_num` and `image_index_on_page` from metadata.

**Image Metadata (use this to insert <img> tags):**
```json
{images_metadata_json}
```

Respond ONLY with the complete HTML code. Do not include any explanations before or after the HTML.
The final HTML should be displayable in Google Chrome and visually resemble the PDF's structure and content.
"""

    user_task_prompt = f"""Please convert the entire PDF (provided as the file input part of this prompt) into a single HTML file.
Follow all instructions in the system prompt precisely, especially regarding:
- Language presentation: {', '.join(target_languages)}.
- Using ONLY pure HTML tags (NO MARKDOWN).
- Directly inserting `<img>` tags using the provided image metadata JSON.
- Including the specified `<head>` content.
- Correctly formatting MathJax equations.
Process all pages of the PDF.
"""

    contents_for_generation = [
        system_instruction_text,
        uploaded_pdf_file_object, # This will be a File object from genai.upload_file
        user_task_prompt
    ]

    print("Sending request to Gemini for HTML generation...")
    try:
        # Ensure response_mime_type is set to text/plain if expecting raw HTML string
        response = text_model.generate_content(
            contents_for_generation,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, # Lower for more deterministic output
                response_mime_type="text/plain" 
            )
        )
        html_output = response.text
    except Exception as e:
        print(f"Error during Gemini HTML generation: {e}")
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            print(f"Prompt feedback: {response.prompt_feedback}")
        return f"<html><head>{head_content}</head><body><h1>Error generating HTML</h1><p>{e}</p></body></html>"

    # Remove potential markdown backticks if model wraps output
    if html_output.strip().startswith("```html"):
        html_output = html_output.split("```html", 1)[-1]
        html_output = html_output.rsplit("```", 1)[0]
    elif html_output.strip().startswith("```"): # Generic markdown block
        html_output = html_output.strip()[3:]
        if html_output.strip().endswith("```"):
            html_output = html_output.strip()[:-3]


    print("HTML generation from Gemini complete.")
    return html_output.strip()

# --- File Upload to Gemini ---
def upload_pdf_to_gemini(pdf_path):
    print(f"Attempting to upload PDF to Gemini: {pdf_path}")
    display_name = f"pdf-conversion-input-{os.path.basename(pdf_path)}"
    
    # Use a try-except block for the upload_file call specifically
    try:
        uploaded_pdf_file = genai.upload_file(path=pdf_path, display_name=display_name, mime_type="application/pdf")
    except Exception as e_upload:
        print(f"Gemini API call to upload_file failed: {e_upload}")
        # You might want to check if this is a google.api_core.exceptions.GoogleAPICallError
        # and inspect e_upload.code() or e_upload.message for more details.
        raise # Re-raise the exception to be handled by the caller

    if not uploaded_pdf_file:
        print("PDF upload failed, genai.upload_file returned None.")
        raise Exception("PDF upload to Gemini returned None")

    print(f"PDF upload initiated. File Name: {uploaded_pdf_file.name}, URI: {uploaded_pdf_file.uri}, State: {uploaded_pdf_file.state.name}")

    polling_interval = 10  # seconds
    max_wait_time = 180    # 3 minutes
    elapsed_time = 0

    while uploaded_pdf_file.state.name == "PROCESSING" and elapsed_time < max_wait_time:
        print(f"Waiting for PDF file '{uploaded_pdf_file.name}' (State: {uploaded_pdf_file.state.name})... waiting {polling_interval}s")
        time.sleep(polling_interval)
        elapsed_time += polling_interval
        try:
            # Re-fetch the file by its name to get updated status
            updated_file_status = genai.get_file(name=uploaded_pdf_file.name)
            if updated_file_status:
                 uploaded_pdf_file = updated_file_status
            else:
                 print(f"Failed to re-fetch status for {uploaded_pdf_file.name}. Aborting wait.")
                 break # Exit loop if status re-fetch fails
        except Exception as e_get_file:
            print(f"Error fetching file status for {uploaded_pdf_file.name}: {e_get_file}. Aborting wait.")
            break


    if uploaded_pdf_file.state.name == "ACTIVE":
        print(f"PDF '{uploaded_pdf_file.name}' is active and ready.")
        return uploaded_pdf_file
    else:
        error_message = f"PDF upload failed or did not become active. Final State: {uploaded_pdf_file.state.name} after {elapsed_time}s."
        if hasattr(uploaded_pdf_file, 'error') and uploaded_pdf_file.error:
            error_details = uploaded_pdf_file.error
            error_message += f" Error during file processing: Code {error_details.code}, Message: {error_details.message}"
        
        # Attempt to delete the failed file
        if uploaded_pdf_file and uploaded_pdf_file.name:
            try:
                print(f"Attempting to delete failed/stuck file: {uploaded_pdf_file.name}")
                genai.delete_file(name=uploaded_pdf_file.name)
                print(f"Cleaned up file {uploaded_pdf_file.name}.")
            except Exception as e_del_fail:
                print(f"Could not delete file {uploaded_pdf_file.name} after failure: {e_del_fail}")
        raise Exception(error_message)


# --- Final HTML Adjustments ---
def finalize_html(html_content, head_content_template, target_languages):
    print("Performing final adjustments on HTML...")

    if not html_content.strip(): # Handle empty or whitespace-only HTML from Gemini
        return f"<!DOCTYPE html><html lang=\"en\"><head>{head_content_template}</head><body><p>Error: Received empty content from the generation model.</p></body></html>"

    # Ensure DOCTYPE is present
    if not html_content.lower().lstrip().startswith("<!doctype html>"):
        html_content = "<!DOCTYPE html>\n" + html_content

    primary_lang_code = target_languages[0].lower().split(' ')[0][:2] if target_languages else 'en'
    # Basic lang codes, can be more specific (e.g., en-US, hi-IN) if needed
    lang_map = {"english": "en", "hindi": "hi", "spanish": "es", "french": "fr", "german": "de", "japanese": "ja", "chinese": "zh"}
    primary_lang_code = lang_map.get(target_languages[0].lower().split('(')[0].strip(), 'en') if target_languages else 'en'


    # Check if <html> tag exists, and if not, wrap content
    if "<html" not in html_content.lower():
        html_content = f"<html lang=\"{primary_lang_code}\">\n<head>\n{head_content_template}\n</head>\n<body>\n{html_content}\n</body>\n</html>"
    else:
        # If <html> exists, ensure lang attribute
        import re
        html_tag_match = re.search(r"<html([^>]*)>", html_content, re.IGNORECASE)
        if html_tag_match:
            attrs = html_tag_match.group(1)
            if f'lang="{primary_lang_code}"' not in attrs.lower() and 'lang=' not in attrs.lower() :
                html_content = re.sub(r"<html([^>]*)>", f"<html lang=\"{primary_lang_code}\"\\1>", html_content, 1, re.IGNORECASE)
            elif 'lang=' in attrs.lower() and f'lang="{primary_lang_code}"' not in attrs.lower():
                 # lang attribute exists but is different, update it.
                 html_content = re.sub(r'lang="[^"]*"', f'lang="{primary_lang_code}"', html_content, 1, re.IGNORECASE)


        # Ensure <head> is present or add it
        if "<head>" not in html_content.lower():
            body_match = re.search(r"<body([^>]*)>", html_content, re.IGNORECASE)
            if body_match:
                html_content = html_content.replace(body_match.group(0), f"<head>\n{head_content_template}\n</head>\n{body_match.group(0)}", 1)
            else: # No body tag either, something is very wrong with model output
                 html_content = f"<head>\n{head_content_template}\n</head>\n<body>\n{html_content}\n</body>" # Wrap if no body found

    print("HTML adjustments complete.")
    return html_content

# List of supported languages for the UI
SUPPORTED_LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi",
    "Marathi": "Marathi",
    "Bengali": "Bengali",
    "Telugu": "Telugu",
    "Tamil": "Tamil",
    "Gujarati": "Gujarati",
    "Urdu": "Urdu",
    "Kannada": "Kannada",
    "Odia": "Odia",
    "Malayalam": "Malayalam",
    "Punjabi": "Punjabi",
    "Assamese": "Assamese",
    "Maithili": "Maithili",
    "Santali": "Santali",
    "Konkani": "Konkani",
    "Kashmiri": "Kashmiri",
    "Dogri": "Dogri",
    "Manipuri": "Manipuri",
    "Bodo": "Bodo",
    "Sindhi": "Sindhi",
    "Sanskrit": "Sanskrit"
}

