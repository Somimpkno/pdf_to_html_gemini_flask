import os
import uuid
import time
import shutil # For disk usage and deleting directories
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import functions from your utility module
from utils.pdf_processor import (
    get_gemini_client_and_models,
    extract_images_and_generate_alt_tags,
    upload_pdf_to_gemini,
    generate_html_from_pdf_gemini,
    finalize_html,
    SUPPORTED_LANGUAGES
)
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'pdf'}
MIN_FREE_SPACE_MB = 70  # MB - Adjust as needed for Render's free tier
                        # (e.g., if free tier gives 500MB, keep 70MB free)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    print(f"Failed to configure Gemini on startup: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_free_space_mb(path='/'):
    """Returns free disk space in MB."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free // (1024 * 1024)
    except Exception as e:
        print(f"Could not get disk space: {e}")
        return float('inf') # Assume enough space if check fails

def cleanup_stale_conversions(current_session_id_to_keep):
    """
    Deletes all subdirectories in UPLOAD_FOLDER and OUTPUT_FOLDER
    except the one matching current_session_id_to_keep.
    """
    print(f"Cleaning up stale data, keeping: {current_session_id_to_keep}")
    folders_to_scan = [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]
    
    for base_folder in folders_to_scan:
        if not os.path.exists(base_folder):
            continue
        for item_name in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item_name)
            # Check if it's a directory and its name looks like our UUID session_id
            if os.path.isdir(item_path) and item_name != current_session_id_to_keep:
                try:
                    # Basic check to ensure it's likely a UUID folder
                    uuid.UUID(item_name, version=4) 
                    print(f"Deleting stale folder: {item_path}")
                    shutil.rmtree(item_path)
                except ValueError:
                    # Not a UUID-named folder, skip
                    print(f"Skipping non-UUID folder: {item_path}")
                except Exception as e:
                    print(f"Error deleting folder {item_path}: {e}")
    print("Cleanup complete.")


@app.route('/')
def index():
    # Store a flag in session if a low storage message was recently shown
    show_storage_warning = session.pop('show_storage_warning', False)
    if show_storage_warning:
        flash(
            f"Storage might be low. If conversion fails, please wait a few minutes and try again, or use a smaller PDF. Current free space: {get_free_space_mb()} MB.",
            'warning'
        )
    return render_template('index.html', supported_languages=SUPPORTED_LANGUAGES)

@app.route('/process', methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    if not file or not allowed_file(file.filename):
        flash('Invalid file type. Only PDF files are allowed.', 'danger')
        return redirect(url_for('index'))

    target_languages = request.form.getlist('target_languages')
    if not target_languages:
        flash('Please select at least one target language.', 'danger')
        return redirect(url_for('index'))

    # --- Start of new logic for cleanup and storage check ---
    # 1. Generate a unique session ID for THIS conversion attempt
    current_processing_session_id = str(uuid.uuid4())
    print(f"Starting new conversion with session ID: {current_processing_session_id}")

    # 2. Cleanup data from ANY OTHER previous conversions
    # This will delete all folders in uploads/ and output/ that don't match current_processing_session_id
    # Note: This is aggressive. If a user had a result page open from a previous conversion, its files will be gone.
    cleanup_stale_conversions(current_processing_session_id) # Pass empty string if you want to clean ALL before creating new one

    # 3. Check available disk space AFTER cleanup
    free_space = get_free_space_mb()
    if free_space < MIN_FREE_SPACE_MB:
        print(f"Low storage: {free_space}MB available, need {MIN_FREE_SPACE_MB}MB.")
        # flash(
        #     f"Server storage is currently low ({free_space}MB available). Please wait a few minutes and try again, or use a smaller PDF. Old files have been cleaned up.",
        #     'warning'
        # )
        session['show_storage_warning'] = True # Set flag to show message on redirect
        return redirect(url_for('index'))
    # --- End of new logic ---

    # Use the generated current_processing_session_id for folder paths
    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_processing_session_id)
    session_output_folder = os.path.join(app.config['OUTPUT_FOLDER'], current_processing_session_id)
    session_extracted_images_folder = os.path.join(session_output_folder, "extracted_images")

    os.makedirs(session_upload_folder, exist_ok=True)
    os.makedirs(session_output_folder, exist_ok=True)
    # extracted_images folder will be created by extract_images_and_generate_alt_tags

    filename = secure_filename(file.filename)
    local_pdf_path = os.path.join(session_upload_folder, filename)
    file.save(local_pdf_path)

    uploaded_gemini_file_object = None

    try:
        vision_model, text_model = get_gemini_client_and_models()
        if not text_model:
            flash("Critical error: Text generation model could not be initialized.", "danger")
            raise Exception("Text model initialization failed.")

        primary_lang_for_alt = target_languages[0] if target_languages else "English"
        
        flash('Step 1: Extracting images and generating alt tags...', 'info')
        images_metadata = extract_images_and_generate_alt_tags(
            local_pdf_path,
            session_extracted_images_folder,
            vision_model,
            primary_lang_for_alt,
            html_image_subfolder="extracted_images"
        )
        flash(f'Successfully processed {len(images_metadata)} images.', 'success')

        flash('Step 2: Uploading PDF to Gemini...', 'info')
        uploaded_gemini_file_object = upload_pdf_to_gemini(local_pdf_path)
        flash(f'PDF "{uploaded_gemini_file_object.name}" uploaded and active.', 'success')
        
        flash('Step 3: Generating HTML from PDF using Gemini...', 'info')
        raw_html_from_gemini = generate_html_from_pdf_gemini(
            text_model,
            uploaded_gemini_file_object,
            images_metadata,
            target_languages
        )

        head_content_template = f"""<meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Content ({', '.join(target_languages)})</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">
    <script>
        window.MathJax = {{
          tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
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
    </style>"""

        flash('Step 4: Finalizing HTML...', 'info')
        final_html_content = finalize_html(raw_html_from_gemini, head_content_template, target_languages)

        final_html_filename = "final_output.html"
        final_html_path = os.path.join(session_output_folder, final_html_filename)
        with open(final_html_path, "w", encoding="utf-8") as f:
            f.write(final_html_content)
        
        flash('Conversion complete!', 'success')
        # Pass current_processing_session_id to the result template
        return render_template('result.html', 
                               success=True, 
                               session_id=current_processing_session_id, 
                               html_filename=final_html_filename,
                               images_folder_exists=os.path.exists(session_extracted_images_folder) and len(os.listdir(session_extracted_images_folder)) > 0
                               )

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        # If an error occurs, the current_processing_session_id folders might have been created.
        # The next run's cleanup_stale_conversions will handle them if this one fails before completion.
        return render_template('result.html', success=False, error_message=str(e), session_id=current_processing_session_id) # Pass session_id for consistency
    
    finally:
        if uploaded_gemini_file_object and hasattr(uploaded_gemini_file_object, 'name'):
            try:
                print(f"Final cleanup: Deleting uploaded PDF '{uploaded_gemini_file_object.name}' from Gemini...")
                genai.delete_file(name=uploaded_gemini_file_object.name)
                print("Uploaded PDF deleted successfully from Gemini.")
            except Exception as e_del:
                print(f"Error during final deletion of uploaded PDF from Gemini: {e_del}")


@app.route('/output/<session_id>/<path:filename>')
def serve_html_file(session_id, filename):
    directory = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    if not os.path.exists(os.path.join(directory, filename)):
        flash('The requested file seems to have been cleaned up. Please try converting again.', 'warning')
        return redirect(url_for('index'))
    return send_from_directory(directory, filename)

@app.route('/output/<session_id>/extracted_images/<path:filename>')
def serve_extracted_image(session_id, filename):
    directory = os.path.join(app.config['OUTPUT_FOLDER'], session_id, 'extracted_images')
    if not os.path.exists(os.path.join(directory, filename)):
        # Don't flash here, as missing images in HTML are less critical than missing HTML itself
        return "Image not found", 404
    return send_from_directory(directory, filename)

@app.route('/download/<session_id>/<filename>')
def download_html_file(session_id, filename):
    directory = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    if not os.path.exists(os.path.join(directory, filename)):
        flash('The requested file seems to have been cleaned up. Please try converting again.', 'warning')
        return redirect(url_for('index'))
    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == '__main__':
    if not os.environ.get("GEMINI_API_KEY"):
        print("FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
    else:
        app.run(debug=True)
