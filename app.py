from flask import Flask, render_template, redirect, url_for, send_file, request, session, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for Flask to save images
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import os
import base64
import io
import uuid
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
app.secret_key = 'data_lab_secret_2026'

# Config - GLOBAL variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'} 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 #20MB

# Custom error handler for >20MB
@app.errorhandler(413)
def too_large(e):
    flash('File size is greater than 20MB')
    return redirect(url_for('upload_file'))

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if file was uploaded
        if 'data_file' not in request.files:
            flash('No file selected.')
            return redirect(request.url)
        
        file = request.files['data_file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load with pandas and store in session
            try:
                if filename.endswith('csv'):
                    df = pd.read_csv(filepath)
                else:  # xlsx
                    df = pd.read_excel(filepath)
                
                # Store file info in session (not full DF - too big)
                session['filename'] = filename
                session['filepath'] = filepath
                session['shape'] = df.shape
                session['columns'] = df.columns.tolist()
                session['numeric_columns'] = df.select_dtypes(include=['number']).columns.tolist()
                
                # Clean up uploaded file (data lives in memory)
                #os.remove(filepath)
                
                flash(f'{filename} uploaded successfully!')
                return redirect(url_for('upload_success'))
                
            except Exception as e:
                flash(f'Error reading file: {str(e)}')
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        else:
            flash('Invalid file type. Use CSV or XLSX.')
    
    return render_template("upload.html")

@app.route("/upload/success")
def upload_success():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))
    
    filename = session['filename']
    return render_template("upload_success.html", filename=filename)

# HELPER: Get DataFrame based on selection (Main vs Custom)
def get_df_dict():
    """Returns a dictionary of all available DFs (Main + Custom)"""
    dfs = {'Main Uploaded File': session.get('filepath')}
    if 'custom_dfs' in session:
        dfs.update(session['custom_dfs'])
    return dfs

def load_selected_df(selection_name):
    """
    Loads a dataframe and sets the first column as index ONLY if 
    it contains valid datetime data. otherwise keeps default index.
    """
    paths = get_df_dict()
    path = paths.get(selection_name)
    
    if not path or not os.path.exists(path):
        return None, None

    # 1. Read as normal dataframe (Columns are just columns)
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception:
        return None, path

    # 2. Strict Check on First Column
    if not df.empty:
        first_col_name = df.columns[0]
        first_col_data = df.iloc[:, 0]
        
        # Check if column is ALREADY datetime (unlikely from CSV, but possible from Excel)
        is_already_datetime = pd.api.types.is_datetime64_any_dtype(first_col_data)
        
        can_be_datetime = False
        
        # If it's Object/String, check if it LOOKS like a date (e.g. "2023-01-01")
        # We STRICTLY avoid numeric types (int/float) to prevent converting ID columns or Sales values
        if not is_already_datetime and first_col_data.dtype == 'object':
            try:
                # Take a sample non-null value
                sample = first_col_data.dropna().iloc[0] if not first_col_data.dropna().empty else None
                if sample:
                    # Attempt conversion on the sample
                    # fuzzy=False ensures we don't accidentally parse "100" as a date easily
                    # We check against common date formats or rely on to_datetime's strictness
                    res = pd.to_datetime(sample, errors='raise')
                    
                    # Double check: Numbers often convert to dates (e.g. 1970-01-01). 
                    # If the original string was purely numeric (e.g. "1001"), reject it.
                    if not str(sample).isdigit(): 
                        can_be_datetime = True
            except:
                can_be_datetime = False

        # 3. Promote ONLY if it passes the strict check
        if is_already_datetime or can_be_datetime:
            # Verify the WHOLE column converts reasonably well (optional but safer)
            # For performance, we assume if sample worked and it's string, it's likely a date index
            df.set_index(first_col_name, inplace=True)
            df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Clean up name if it was auto-generated
            if "Unnamed" in str(first_col_name):
                df.index.name = None

    return df, path

# 3.1 Modifications menu
@app.route("/modifications")
def modifications_menu():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))
    return render_template("modifications/modifications_index.html")

# 3.1.1 Sorting (UPDATED with Ascending/Descending)
@app.route("/modifications/sorting", methods=["GET", "POST"])
def sorting():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('modifications_menu'))

    columns = df.columns.tolist()
    head_html = None
    available_dfs = get_df_dict().keys()

    if request.method == "POST":
        action = request.form.get('action') 

        if action == 'sort':
            sort_col = request.form.get('sort_col')
            sort_order = request.form.get('sort_order') # 'asc' or 'desc'
            
            # Determine Boolean for ascending
            is_ascending = True if sort_order == 'asc' else False

            if sort_col in columns:
                # Apply Sorting with Order
                df_sorted = df.sort_values(by=sort_col, ascending=is_ascending)
                
                # Overwrite file
                if path.endswith('.csv'):
                    df_sorted.to_csv(path, index=False)
                else:
                    df_sorted.to_excel(path, index=False)
                
                head_html = df_sorted.head().to_html(classes='table table-striped', float_format='%.3f')
                flash(f"Sorted '{current_selection}' by '{sort_col}' ({'Ascending' if is_ascending else 'Descending'})!")

    return render_template("modifications/sorting.html", 
                           columns=columns, 
                           head_html=head_html,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# Route for downloading the CURRENT state of the file
@app.route("/download_current")
def download_current():
    if 'filepath' not in session:
        return redirect(url_for('upload_file'))
    return send_file(session['filepath'], as_attachment=True, download_name=session['filename'])

# 3.1.2 Filtering (FIXED: Preserves Index)
@app.route("/modifications/filtering", methods=["GET", "POST"])
def filtering():
    if 'filename' not in session:
        flash('No file uploaded.')
        return redirect(url_for('upload_file'))

    if 'custom_dfs' not in session:
        session['custom_dfs'] = {}

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('modifications_menu'))
    
    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()

    preview_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')
        
        if action == "apply_filter":
            # ... [Validation Logic Same as Before] ...
            df_name = request.form.get('df_name')
            target_col = request.form.get('target_col')
            operator = request.form.get('operator')
            filter_val = request.form.get('filter_val')

            # ... [Validation Checks] ...
            if not df_name or df_name in session['custom_dfs']:
                flash("Invalid name or name already exists.")
            elif target_col not in columns:
                flash("Invalid column selected.")
            else:
                try:
                    # Apply Logic (Same as before)
                    if df[target_col].dtype.kind in 'bifc': # Numeric/Boolean
                        filter_val = float(filter_val)
                        if operator == '<': new_df = df[df[target_col] < filter_val]
                        elif operator == '<=': new_df = df[df[target_col] <= filter_val]
                        elif operator == '>': new_df = df[df[target_col] > filter_val]
                        elif operator == '>=': new_df = df[df[target_col] >= filter_val]
                        elif operator == '==': new_df = df[df[target_col] == filter_val]
                        elif operator == '!=': new_df = df[df[target_col] != filter_val]
                    else: 
                        # String comparison
                        if operator == '==': new_df = df[df[target_col].astype(str) == str(filter_val)]
                        elif operator == '!=': new_df = df[df[target_col].astype(str) != str(filter_val)]
                        else:
                            flash("Operator not supported for non-numeric columns.")
                            return redirect(request.url)

                    # --- CRITICAL FIX: PRESERVE INDEX ---
                    filename = f"custom_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    # Check if index is special (Datetime or named)
                    # Ideally, we ALWAYS save index=True for consistency in this app,
                    # because load_selected_df is smart enough to handle Unnamed: 0 vs Real Index now.
                    # BUT, if the index is just RangeIndex (0, 1, 2...), saving it creates an annoying 'Unnamed: 0' column later.
                    
                    has_special_index = isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None
                    
                    # Better Logic: If it WAS loaded with an index (via load_selected_df), we should save it.
                    # Since we don't track *how* it was loaded easily, let's rely on type.
                    
                    if has_special_index:
                         new_df.to_csv(save_path, index=True)
                    else:
                         # Check if it's just a default range index (0, 1, 2...)
                         if pd.api.types.is_integer_dtype(df.index) and df.index.start == 0 and df.index.step == 1:
                             new_df.to_csv(save_path, index=False) # Don't save 0,1,2...
                         else:
                             new_df.to_csv(save_path, index=True) # Save meaningful index (IDs, etc)

                    # Update Session
                    session['custom_dfs'][df_name] = save_path
                    session.modified = True
                    
                    new_df_name = df_name
                    preview_html = new_df.head().to_html(classes='table table-striped', float_format='%.3f')
                    flash(f"Filter applied on '{current_selection}'. New DataFrame '{df_name}' created!")

                except Exception as e:
                    flash(f"Error applying filter: {str(e)}")

    return render_template("modifications/filtering.html", 
                           columns=columns, 
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

@app.route("/download_custom/<name>")
def download_custom(name):
    if 'custom_dfs' in session and name in session['custom_dfs']:
        return send_file(session['custom_dfs'][name], as_attachment=True, download_name=f"{name}.csv")
    flash("File not found.")
    return redirect(url_for('filtering'))

# 3.1.3 Change Indices
@app.route("/modifications/change_indices", methods=["GET", "POST"])
def change_indices():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Determine Selection (Default Main)
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    # 2. Load DF
    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('modifications_menu'))

    columns = df.columns.tolist()
    head_html = None
    available_dfs = get_df_dict().keys()

    if request.method == "POST":
        action = request.form.get('action')

        if action == 'set_index':
            index_col = request.form.get('index_col')
            
            if index_col in columns:
                try:
                    # Apply Index
                    df.set_index(index_col, inplace=True)
                    
                    # Save (Index=True to keep the new index)
                    if path.endswith('.csv'):
                        df.to_csv(path, index=True)
                    else:
                        df.to_excel(path, index=True)
                    
                    head_html = df.head().to_html(classes='table table-striped', float_format='%.3f')
                    flash(f"Index set to '{index_col}' successfully!")
                    
                    # Reload columns (since index is no longer a column)
                    columns = df.reset_index().columns.tolist() # Just for dropdown logic if needed next
                    
                except Exception as e:
                    flash(f"Error setting index: {str(e)}")

    return render_template("modifications/change_indices.html", 
                           columns=columns, 
                           head_html=head_html,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.1.4 Segmentation
@app.route("/modifications/segmentation", methods=["GET", "POST"])
def segmentation():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Determine Selection
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST" and 'selected_df' in request.form:
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    # 2. Load DF
    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('modifications_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    
    preview_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'segment':
            # Inputs
            new_name = request.form.get('new_name')
            segment_col = request.form.get('segment_col')
            agg_func = request.form.get('agg_func')
            agg_cols = request.form.getlist('agg_cols') # List of checked columns

            # Validation
            if not new_name or new_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif not segment_col or not agg_cols:
                flash("Please select a segmentation column and at least one aggregation column.")
            else:
                try:
                    # Perform Segmentation
                    # Note: We reset_index() so the segment_col becomes a regular column again in the new DF
                    grouped_df = df.groupby(segment_col)[agg_cols].agg(agg_func).reset_index()
                    
                    # Save New DF
                    filename = f"custom_seg_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    grouped_df.to_csv(save_path, index=False)

                    # Update Session
                    if 'custom_dfs' not in session: session['custom_dfs'] = {}
                    session['custom_dfs'][new_name] = save_path
                    session.modified = True
                    
                    new_df_name = new_name
                    preview_html = grouped_df.head().to_html(classes='table table-striped', float_format='%.3f')
                    flash(f"Segmentation successful! Created '{new_name}'.")

                except Exception as e:
                    flash(f"Error during segmentation: {str(e)} (Ensure you selected numeric columns for math functions)")

    return render_template("modifications/segmentation.html", 
                           columns=columns, 
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2 Time Series Modifications Menu
@app.route("/time_series")
def time_series_menu():
    if 'filename' not in session:
        flash('No file uploaded.')
        return redirect(url_for('upload_file'))
    return render_template("time_series/time_series_index.html")

# 3.2.1 Convert to Date Indices (UPDATED with Custom DF Creation)
@app.route("/time_series/to_date_index", methods=["GET", "POST"])
def convert_date_index():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    
    head_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')

        if action == 'convert':
            df_name = request.form.get('df_name')
            date_col = request.form.get('date_col')
            
            if not df_name or df_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif date_col in columns:
                try:
                    # Convert to Datetime
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    if df[date_col].isnull().any():
                        flash(f"Warning: Some rows in '{date_col}' became NaT (Not a Time).")
                    
                    # Set Index
                    df.set_index(date_col, inplace=True)
                    
                    # SAVE AS NEW CUSTOM DF
                    filename = f"custom_index_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    df.to_csv(save_path, index=True) # Keep the index in the saved file
                    
                    # Update Session
                    if 'custom_dfs' not in session: session['custom_dfs'] = {}
                    session['custom_dfs'][df_name] = save_path
                    session.modified = True
                    
                    new_df_name = df_name
                    head_html = df.head().to_html(classes='table table-striped', float_format='%.3f')
                    flash(f"Success! Created '{df_name}' with '{date_col}' as index.")
                    
                    # Reload columns for display (index is gone from cols)
                    columns = df.reset_index().columns.tolist()

                except Exception as e:
                    flash(f"Error converting column: {str(e)}")

    return render_template("time_series/to_date_index.html", 
                           columns=columns, 
                           head_html=head_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2.2 Date Filtering (FIXED & DEBUGGED)
@app.route("/time_series/date_filtering", methods=["GET", "POST"])
def date_filtering():
    # --- 1. SETUP ---
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    # --- 2. DETECT OPTIONS ---
    date_options = []
    
    # Index Check
    try:
        # Check if index is datetime-like
        idx_test = pd.to_datetime(df.index, errors='coerce')
        if not idx_test.isnull().all():
            date_options.insert(0, {'value': '__index__', 'label': f"Index ({df.index.name or 'Date Index'})"})
    except:
        pass

    # Column Check
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_options.append({'value': col, 'label': f"Column: {col}"})
        elif df[col].dtype == 'object':
            first = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first:
                try:
                    pd.to_datetime(first)
                    date_options.append({'value': col, 'label': f"Column: {col} (detected)"})
                except: pass

    available_dfs = get_df_dict().keys()
    preview_html = None
    new_df_name = None

    # --- 3. FILTER LOGIC ---
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'filter':
            df_name = request.form.get('df_name')
            target = request.form.get('target') 
            operator = request.form.get('operator')
            filter_val = request.form.get('filter_val')

            if not target:
                flash("Error: Please select a date column.")
            else:
                try:
                    # 1. Parse User Input
                    user_dt = pd.to_datetime(filter_val)
                    
                    # 2. Get Data as a Series (Unified Method)
                    if target == '__index__':
                        # Create a Series from the index, using the same index for alignment
                        data_series = pd.Series(df.index, index=df.index)
                    else:
                        data_series = df[target]

                    # 3. Force Convert to Datetime (Coerce errors)
                    data_series = pd.to_datetime(data_series, errors='coerce')

                    # 4. Normalize if User Input is "Date Only" (00:00:00)
                    if user_dt.time() == pd.Timestamp("00:00:00").time():
                        # Access .dt accessor safely
                        data_series = data_series.dt.normalize()

                    # 5. Create Boolean Mask
                    if operator == '==': mask = (data_series == user_dt)
                    elif operator == '!=': mask = (data_series != user_dt)
                    elif operator == '>': mask = (data_series > user_dt)
                    elif operator == '>=': mask = (data_series >= user_dt)
                    elif operator == '<': mask = (data_series < user_dt)
                    elif operator == '<=': mask = (data_series <= user_dt)
                    else: mask = pd.Series([False] * len(df), index=df.index)

                    # 6. Apply Mask
                    new_df = df[mask]

                    # 7. Check Result
                    if new_df.empty:
                        flash(f"Warning: Filter resulted in 0 rows. (Input: {user_dt.date()})")
                    else:
                        flash(f"Success! Created '{df_name}' with {len(new_df)} rows.")

                    # 8. Save (Even if empty, to allow user verification)
                    filename = f"custom_date_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    new_df.to_csv(save_path, index=True)

                    if 'custom_dfs' not in session: session['custom_dfs'] = {}
                    session['custom_dfs'][df_name] = save_path
                    session.modified = True
                    
                    new_df_name = df_name
                    preview_html = new_df.head().to_html(classes='table table-striped', float_format='%.3f')

                except Exception as e:
                    # Capture the specific error
                    flash(f"Critical Error: {str(e)}")
                    print(f"DEBUG ERROR: {e}") # Check your console/terminal for this if flash fails

    return render_template("time_series/date_filtering.html", 
                           date_options=date_options, 
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2.3 Date Segmentation (COMPLETE WORKING - FULL TABLE)
@app.route("/time_series/date_segmentation", methods=["GET", "POST"])
def date_segmentation():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    # Detect Date Options
    date_options = []
    
    # Index
    try:
        temp_idx = pd.to_datetime(df.index, errors='coerce')
        if not temp_idx.isnull().all():
            date_options.insert(0, {'value': '__index__', 'label': f"Index ({df.index.name or 'Date Index'})"})
    except:
        pass

    # Columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_options.append({'value': col, 'label': f"Column: {col}"})
        elif df[col].dtype == 'object':
            try:
                first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if first_val:
                    pd.to_datetime(first_val)
                    date_options.append({'value': col, 'label': f"Column: {col} (detected)"})
            except:
                pass

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    available_dfs = get_df_dict().keys()
    preview_html = None
    new_df_name = None

    if request.method == "POST":
        if request.form.get('action') == 'segment':
            df_name = request.form.get('df_name')
            date_target = request.form.get('date_target')
            segment_type = request.form.get('segment_type')
            agg_func = request.form.get('agg_func')
            agg_cols = request.form.getlist('agg_cols')

            if not df_name or df_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif not date_target or not agg_cols:
                flash("Please select date column and columns to aggregate.")
            else:
                try:
                    # STEP 1: PREPARE DATE DATA
                    valid_rows = None
                    grouper = None
                    
                    if date_target == '__index__':
                        # INDEX: Direct DatetimeIndex access
                        date_data = pd.to_datetime(df.index, errors='coerce')
                        valid_rows = date_data.notna()
                        
                        if segment_type == 'day':
                            grouper = date_data[valid_rows].day
                        elif segment_type == 'month':
                            grouper = date_data[valid_rows].month  # 1-12
                        elif segment_type == 'year':
                            grouper = date_data[valid_rows].year
                            
                    else:
                        # COLUMN: Series with .dt
                        date_series = pd.to_datetime(df[date_target], errors='coerce')
                        valid_rows = date_series.notna()
                        
                        if segment_type == 'day':
                            grouper = date_series[valid_rows].dt.day
                        elif segment_type == 'month':
                            grouper = date_series[valid_rows].dt.month  # 1-12
                        elif segment_type == 'year':
                            grouper = date_series[valid_rows].dt.year

                    if not valid_rows.any():
                        flash("No valid dates found.")
                    else:
                        # STEP 2: PREPARE AGG DATA
                        agg_data = df.loc[valid_rows, agg_cols].copy()
                        
                        # Force numeric conversion
                        for col in agg_cols:
                            agg_data[col] = pd.to_numeric(agg_data[col], errors='coerce')
                        
                        # STEP 3: SIMPLIFIED GROUPBY (Direct approach)
                        # Create temp dataframe with grouper + agg columns
                        temp_df = pd.DataFrame({'grouper': grouper})
                        for col in agg_cols:
                            temp_df[col] = agg_data[col].values
                        
                        # Groupby with as_index=False to avoid index issues
                        grouped_df = temp_df.groupby('grouper', as_index=False).agg(agg_func).round(3)
                        
                        # Rename grouper column based on segment_type
                        if segment_type == 'month':
                            grouped_df = grouped_df.rename(columns={'grouper': 'Month'})
                        elif segment_type == 'day':
                            grouped_df = grouped_df.rename(columns={'grouper': 'Day'})
                        elif segment_type == 'year':
                            grouped_df = grouped_df.rename(columns={'grouper': 'Year'})

                        # STEP 4: SAVE & SHOW FULL TABLE
                        filename = f"custom_seg_{uuid.uuid4().hex[:8]}.csv"
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        grouped_df.to_csv(save_path, index=False)

                        if 'custom_dfs' not in session:
                            session['custom_dfs'] = {}
                        session['custom_dfs'][df_name] = save_path
                        session.modified = True

                        new_df_name = df_name
                        preview_html = grouped_df.to_html(classes='table table-striped', float_format='%.3f')  # FULL TABLE
                        flash(f"✅ Segmented by {segment_type}! '{df_name}' created ({len(grouped_df)} groups)")

                except Exception as e:
                    flash(f"❌ Error: {str(e)}")

    return render_template("time_series/date_segmentation.html", 
                           date_options=date_options,
                           numeric_cols=numeric_cols,
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2.4 Moving Average
@app.route("/time_series/moving_average", methods=["GET", "POST"])
def moving_average():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Load Selection
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    # 2. Get Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    available_dfs = get_df_dict().keys()
    
    preview_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'smooth':
            df_name = request.form.get('df_name')
            target_col = request.form.get('target_col')
            window_size = request.form.get('window_size')

            if not df_name or df_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif not target_col or not window_size:
                flash("Please provide column and window size.")
            else:
                try:
                    # Validate Window Size
                    window = int(window_size)
                    if window <= 0:
                        raise ValueError("Window size must be positive.")

                    # Calculate Moving Average
                    # We create a new column name like "Sales_MA_7"
                    ma_col_name = f"{target_col}_MA_{window}"
                    
                    # Create a copy to avoid SettingWithCopy warnings on slices
                    new_df = df.copy()
                    
                    # Apply Rolling Mean
                    new_df[ma_col_name] = new_df[target_col].rolling(window=window).mean()

                    # Save
                    filename = f"custom_ma_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    # Save index=True if it has a named index, otherwise False? 
                    # Usually safest to keep index=True for time series consistency
                    new_df.to_csv(save_path, index=True)

                    if 'custom_dfs' not in session: session['custom_dfs'] = {}
                    session['custom_dfs'][df_name] = save_path
                    session.modified = True
                    
                    new_df_name = df_name
                    
                    # Preview: Show head(window + 5) so user sees the first calculated values (after NaNs)
                    # We'll actually slice slightly to show the transition from NaN to Value if possible,
                    # but head(window + 5) is requested specifically.
                    preview_html = new_df.head(window + 5).to_html(classes='table table-striped', float_format='%.3f')
                    
                    flash(f"Smoothed '{target_col}' with window {window}! Created '{df_name}'.")

                except ValueError:
                    flash("Window size must be a valid integer.")
                except Exception as e:
                    flash(f"Error: {str(e)}")

    return render_template("time_series/moving_average.html", 
                           numeric_cols=numeric_cols,
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2.5 Percent Change
@app.route("/time_series/percent_change", methods=["GET", "POST"])
def percent_change():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Load Selection
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    # 2. Get Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    available_dfs = get_df_dict().keys()
    
    preview_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'calculate':
            df_name = request.form.get('df_name')
            target_col = request.form.get('target_col')

            if not df_name or df_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif not target_col:
                flash("Please select a column.")
            else:
                try:
                    # Create copy
                    new_df = df.copy()
                    
                    # Calculate % Change
                    pct_col_name = f"{target_col}_Pct_Change"
                    
                    # pct_change() gives decimal (e.g. 0.05 for 5%), so multiply by 100
                    new_df[pct_col_name] = new_df[target_col].pct_change() * 100

                    # Save
                    filename = f"custom_pct_{uuid.uuid4().hex[:8]}.csv"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    new_df.to_csv(save_path, index=True)

                    if 'custom_dfs' not in session: session['custom_dfs'] = {}
                    session['custom_dfs'][df_name] = save_path
                    session.modified = True
                    
                    new_df_name = df_name
                    
                    # Preview (First 10 rows)
                    # Note: Row 0 will be NaN for pct_change
                    preview_html = new_df.head(10).to_html(classes='table table-striped', float_format='%.2f')
                    
                    flash(f"Calculated Percent Change for '{target_col}'! Created '{df_name}'.")

                except Exception as e:
                    flash(f"Error: {str(e)}")

    return render_template("time_series/percent_change.html", 
                           numeric_cols=numeric_cols,
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.2.6 Resample Data
@app.route("/time_series/resample", methods=["GET", "POST"])
def resample_data():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Load Selection
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('time_series_menu'))

    # Check if we have Datetime Index for resampling
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
    # Resampling options
    resample_rules = [
        ('D', 'Daily'),
        ('W', 'Weekly (Sunday)'),
        ('W-MON', 'Weekly (Monday)'),
        ('ME', 'Month End'),
        ('MS', 'Month Start'),
        ('QE', 'Quarter End'),
        ('QS', 'Quarter Start'),
        ('YE', 'Year End'),
        ('YS', 'Year Start'),
        ('H', 'Hourly')
    ]
    
    agg_functions = ['mean', 'sum', 'min', 'max', 'median', 'count']
    
    available_dfs = get_df_dict().keys()
    preview_html = None
    new_df_name = None

    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'resample':
            df_name = request.form.get('df_name')
            rule = request.form.get('rule')
            agg_func = request.form.get('agg_func')

            if not df_name or df_name in session.get('custom_dfs', {}):
                flash("Invalid name or name already exists.")
            elif not has_datetime_index:
                flash("Resampling requires a Datetime Index. Use 'Convert to Date Indices' (3.2.1) first.")
            elif not rule or not agg_func:
                flash("Please select resampling rule and aggregation function.")
            else:
                try:
                    # Select only numeric columns automatically
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if not numeric_cols:
                        flash("No numeric columns found for resampling.")
                    else:
                        # Resample ALL numeric columns
                        resampled = df[numeric_cols].resample(rule).agg(agg_func)
                        
                        # Reset index to make dates a column (easier for preview/download)
                        #resampled = resampled.reset_index()
                        
                        # Save
                        filename = f"custom_resample_{uuid.uuid4().hex[:8]}.csv"
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        resampled.to_csv(save_path, index=True)

                        if 'custom_dfs' not in session: 
                            session['custom_dfs'] = {}
                        session['custom_dfs'][df_name] = save_path
                        session.modified = True
                        
                        new_df_name = df_name
                        preview_html = resampled.head().to_html(classes='table table-striped', float_format='%.3f')
                        flash(f"Resampled to {resample_rules[[r[0] for r in resample_rules].index(rule)][1]} using {agg_func}! Created '{df_name}'.")

                except Exception as e:
                    flash(f"Error during resampling: {str(e)}")

    return render_template("time_series/resample.html", 
                           has_datetime_index=has_datetime_index,
                           resample_rules=resample_rules,
                           agg_functions=agg_functions,
                           preview_html=preview_html,
                           new_df_name=new_df_name,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

#3.4 Visualization
@app.route("/visualization")
def visualization_menu():
    if 'filename' not in session:
        flash('Please upload a file first.')
        return redirect(url_for('upload_file'))
    return render_template("visualization/visualization_menu.html")

# 3.4.1 Column Chart
@app.route("/visualization/column_chart", methods=["GET", "POST"])
def column_chart():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    # 1. Load Selection
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    chart_image = None
    
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'create':
            try:
                # Inputs
                x_col = request.form.get('xaxis')
                y_col = request.form.get('yaxis')
                
                x_label = request.form.get('xlabel') or x_col
                y_label = request.form.get('ylabel') or y_col
                title = request.form.get('title')
                
                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)
                
                color = request.form.get('color') or None # None uses default
                grid_opt = request.form.get('grid') # 'on' or 'off' (or None)

                if not x_col or not y_col:
                    flash("Please select both X and Y axes.")
                else:
                    # Plotting
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create Bar Chart
                    ax.bar(df[x_col].astype(str), df[y_col], color=color) # Ensure X is string for categorical plotting
                    
                    # Styling
                    ax.set_xlabel(x_label, fontsize=14)
                    ax.set_ylabel(y_label, fontsize=14)
                    if title:
                        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
                    
                    ax.tick_params(axis='x', rotation=x_rot)
                    ax.tick_params(axis='y', rotation=y_rot)
                    
                    if grid_opt == 'on':
                        ax.grid(True, linestyle='--', alpha=0.7)
                    else:
                        ax.grid(False)

                    # Layout
                    plt.tight_layout()

                    # Save Image
                    filename = f"chart_{uuid.uuid4().hex[:8]}.jpeg"
                    save_path = os.path.join('static', 'charts', filename) # Save to static/charts
                    
                    # Ensure directory exists
                    os.makedirs(os.path.join('static', 'charts'), exist_ok=True)
                    
                    plt.savefig(save_path, format='jpeg', dpi=300)
                    plt.close(fig) # Close to free memory

                    chart_image = filename # Pass filename to template
                    flash("Chart created successfully!")

            except Exception as e:
                flash(f"Error creating chart: {str(e)}")

    return render_template("visualization/column_chart.html", 
                           columns=columns,
                           chart_image=chart_image,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.4.2 Grouped Column Chart
@app.route("/visualization/grouped_column_chart", methods=["GET", "POST"])
def grouped_column_chart():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    if hasattr(df.index, 'names') and df.index.names[0] is not None:
        columns.extend([n for n in df.index.names if n])

    available_dfs = get_df_dict().keys()
    chart_image = None
    
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'create':
            try:
                x_col = request.form.get('xaxis')
                y_col = request.form.get('yaxis')
                
                x_label = request.form.get('xlabel') or (x_col if x_col else "Categories")
                y_label = request.form.get('ylabel') or (y_col if y_col else "Values")
                title = request.form.get('title')
                
                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)
                
                color_input = request.form.get('color')
                colors = [c.strip() for c in color_input.split(',')] if color_input else None
                
                grid_opt = request.form.get('grid')

                fig, ax = plt.subplots(figsize=(10, 6))
                
                # --- Simplified Logic (Unstack removed) ---
                plot_df = df.copy()
                
                if x_col:
                    # Set X column as index so pandas plots the remaining columns as grouped bars
                    plot_df = plot_df.set_index(x_col)
                    
                    if y_col:
                        # If a specific Y column is selected, only plot that one
                        plot_df = plot_df[[y_col]]
                    else:
                        # If Y is left blank, plot all numeric columns as grouped bars
                        plot_df = plot_df.select_dtypes(include='number')

                # Plotting
                if colors:
                    plot_df.plot(kind='bar', ax=ax, color=colors)
                else:
                    plot_df.plot(kind='bar', ax=ax)
                
                # Styling
                ax.set_xlabel(x_label, fontsize=14)
                ax.set_ylabel(y_label, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
                
                ax.tick_params(axis='x', rotation=x_rot)
                ax.tick_params(axis='y', rotation=y_rot)
                
                if grid_opt == 'on':
                    ax.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax.grid(False)

                try:
                    plt.tight_layout()
                except Exception:
                    pass 

                # Save Image
                filename = f"grouped_chart_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join('static', 'charts', filename) 
                
                os.makedirs(os.path.join('static', 'charts'), exist_ok=True)
                
                plt.savefig(save_path, format='jpeg', dpi=300)
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error creating chart: {str(e)}", "error")

    return render_template("visualization/grouped_column_chart.html", 
                           columns=columns,
                           chart_image=chart_image,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.4.3 Stacked Column Chart
@app.route("/visualization/stacked_column_chart", methods=["GET", "POST"])
def stacked_column_chart():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    if hasattr(df.index, 'names') and df.index.names[0] is not None:
        columns.extend([n for n in df.index.names if n])

    available_dfs = get_df_dict().keys()
    chart_image = None
    
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'create':
            try:
                x_col = request.form.get('xaxis')
                y_col = request.form.get('yaxis')
                
                x_label = request.form.get('xlabel') or (x_col if x_col else "Categories")
                y_label = request.form.get('ylabel') or (y_col if y_col else "Values")
                title = request.form.get('title')
                
                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)
                
                color_input = request.form.get('color')
                colors = [c.strip() for c in color_input.split(',')] if color_input else None
                
                grid_opt = request.form.get('grid')

                # Prepare data
                plot_df = df.copy()

                if x_col:
                    plot_df = plot_df.set_index(x_col)

                if y_col:
                    # If Y selected, we use that as a single series
                    # (still technically stacked, but single stack)
                    plot_df = plot_df[[y_col]]
                else:
                    # If Y not selected, stack all numeric columns
                    plot_df = plot_df.select_dtypes(include='number')

                if plot_df.empty:
                    raise ValueError("No numeric columns available to plot. Please check your grouped data.")

                fig, ax = plt.subplots(figsize=(10, 6))

                if colors:
                    plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors)
                else:
                    plot_df.plot(kind='bar', stacked=True, ax=ax)

                ax.set_xlabel(x_label, fontsize=14)
                ax.set_ylabel(y_label, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

                ax.tick_params(axis='x', rotation=x_rot)
                ax.tick_params(axis='y', rotation=y_rot)

                if grid_opt == 'on':
                    ax.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax.grid(False)

                try:
                    plt.tight_layout()
                except Exception:
                    pass

                filename = f"stacked_chart_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join('static', 'charts', filename)
                os.makedirs(os.path.join('static', 'charts'), exist_ok=True)

                plt.savefig(save_path, format='jpeg', dpi=300)
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error creating chart: {str(e)}", "error")

    return render_template("visualization/stacked_column_chart.html",
                           columns=columns,
                           chart_image=chart_image,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.4.4 Line Chart (single column, sns.lineplot)
@app.route("/visualization/line_chart", methods=["GET", "POST"])
def line_chart():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    chart_image = None
    
    if request.method == "POST":
        action = request.form.get('action')
        
        if action == 'create':
            try:
                line_col = request.form.get('line_col')  # single column input

                if not line_col:
                    raise ValueError("Please select a column to plot.")

                x_label = request.form.get('xlabel') or "Index"
                y_label = request.form.get('ylabel') or line_col
                title = request.form.get('title')
                
                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)
                
                color_input = request.form.get('color')
                color = color_input.strip() if color_input else None

                linestyle = request.form.get('linestyle') or 'solid'
                grid_opt = request.form.get('grid')

                # Prepare data
                plot_df = df[[line_col]].reset_index(drop=False)
                plot_df.rename(columns={plot_df.columns[0]: "index"}, inplace=True)

                fig, ax = plt.subplots(figsize=(10, 6))

                # sns.lineplot
                line_kws = {
                    "data": plot_df,
                    "x": "index",
                    "y": line_col,
                    "ax": ax,
                    "linestyle": linestyle
                }
                if color:
                    line_kws["color"] = color

                sns.lineplot(**line_kws)

                ax.set_xlabel(x_label, fontsize=14)
                ax.set_ylabel(y_label, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

                ax.tick_params(axis='x', rotation=x_rot)
                ax.tick_params(axis='y', rotation=y_rot)

                if grid_opt == 'on':
                    ax.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax.grid(False)

                try:
                    plt.tight_layout()
                except Exception:
                    pass

                filename = f"line_chart_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join('static', 'charts', filename)
                os.makedirs(os.path.join('static', 'charts'), exist_ok=True)

                plt.savefig(save_path, format='jpeg', dpi=300)
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error creating chart: {str(e)}", "error")

    return render_template("visualization/line_chart.html",
                           columns=columns,
                           chart_image=chart_image,
                           available_dfs=available_dfs,
                           current_selection=current_selection)

# 3.4.5 Histogram - COMPLETE WORKING VERSION
@app.route("/visualization/histogram", methods=["GET", "POST"])
def histogram():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    chart_image = None

    if request.method == "POST":
        if request.form.get('action') == 'create':
            try:
                col = request.form.get('histogram_col')
                
                if not col or col not in df.columns:
                    raise ValueError("Please select a valid column.")

                series = df[col].dropna()
                if series.empty or not pd.api.types.is_numeric_dtype(series):
                    raise ValueError("Selected column must have numeric data.")

                # Form inputs with safe defaults
                xlabel = request.form.get('xlabel', 'Frequency')
                ylabel = request.form.get('ylabel', col)
                title = request.form.get('title', '')
                
                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)
                
                color = request.form.get('color') or None
                bins = int(request.form.get('bins') or 20)
                kde_opt = request.form.get('kde') == 'on'
                grid_opt = request.form.get('grid') == 'on'

                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # seaborn histplot
                sns.histplot(
                    data=series, 
                    bins=bins, 
                    kde=kde_opt, 
                    color=color, 
                    ax=ax
                )

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

                ax.tick_params(axis="x", rotation=x_rot)
                ax.tick_params(axis="y", rotation=y_rot)

                if grid_opt:
                    ax.grid(True, linestyle="--", alpha=0.7)

                plt.tight_layout()

                # Save chart
                filename = f"histogram_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join("static", "charts", filename)
                os.makedirs(os.path.join("static", "charts"), exist_ok=True)

                plt.savefig(save_path, format="jpeg", dpi=300, bbox_inches='tight')
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "visualization/histogram.html",
        columns=columns,
        chart_image=chart_image,
        available_dfs=available_dfs,
        current_selection=current_selection,
    )

# 3.4.6 Scatter Chart
@app.route("/visualization/scatter_chart", methods=["GET", "POST"])
def scatter_chart():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    # Only numeric columns for x/y
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    available_dfs = get_df_dict().keys()
    chart_image = None

    if request.method == "POST":
        if request.form.get('action') == 'create':
            try:
                x_col = request.form.get('xaxis')
                y_col = request.form.get('yaxis')

                if not x_col or not y_col:
                    raise ValueError("Please select both X-axis and Y-axis columns.")

                if x_col not in numeric_cols or y_col not in numeric_cols:
                    raise ValueError("X and Y axes must be numeric columns.")

                x = df[x_col].dropna()
                y = df[y_col].dropna()
                # align indexes
                xy = pd.concat([x, y], axis=1).dropna()
                x = xy[x_col]
                y = xy[y_col]

                xlabel = request.form.get('xlabel') or x_col
                ylabel = request.form.get('ylabel') or y_col
                title = request.form.get('title')

                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)

                color = (request.form.get('color') or "").strip() or None
                grid_opt = request.form.get('grid') == 'on'

                marker = request.form.get('marker') or 'o'
                alpha_val = request.form.get('alpha')
                try:
                    alpha = float(alpha_val) if alpha_val else 1.0
                    if not (0 <= alpha <= 1):
                        raise ValueError
                except ValueError:
                    raise ValueError("Alpha must be a number between 0 and 1.")

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.scatter(
                    x,
                    y,
                    c=color,
                    alpha=alpha,
                    marker=marker
                )

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

                ax.tick_params(axis='x', rotation=x_rot)
                ax.tick_params(axis='y', rotation=y_rot)

                if grid_opt:
                    ax.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()

                filename = f"scatter_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join("static", "charts", filename)
                os.makedirs(os.path.join("static", "charts"), exist_ok=True)

                plt.savefig(save_path, format="jpeg", dpi=300, bbox_inches='tight')
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "visualization/scatter_chart.html",
        numeric_cols=numeric_cols,
        chart_image=chart_image,
        available_dfs=available_dfs,
        current_selection=current_selection,
    )

# 3.4.7 Box Plot
@app.route("/visualization/box_plot", methods=["GET", "POST"])
def box_plot():
    if 'filename' not in session:
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('visualization_menu'))

    columns = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    chart_image = None

    if request.method == "POST":
        if request.form.get('action') == 'create':
            try:
                x_col = request.form.get('xaxis') or None
                y_col = request.form.get('yaxis') or None

                if not x_col and not y_col:
                    raise ValueError("Please select at least one of X-axis or Y-axis.")

                # Validate numeric requirement for the continuous axis
                # If only one is selected, it must be numeric
                if x_col and not y_col:
                    if not pd.api.types.is_numeric_dtype(df[x_col]):
                        raise ValueError(f"For a horizontal box plot, '{x_col}' must be numeric.")
                elif y_col and not x_col:
                    if not pd.api.types.is_numeric_dtype(df[y_col]):
                        raise ValueError(f"For a vertical box plot, '{y_col}' must be numeric.")
                elif x_col and y_col:
                    # If both are selected, at least one must be numeric for seaborn to work correctly
                    x_num = pd.api.types.is_numeric_dtype(df[x_col])
                    y_num = pd.api.types.is_numeric_dtype(df[y_col])
                    if not x_num and not y_num:
                        raise ValueError("When selecting both axes, at least one must be a numerical column.")

                xlabel = request.form.get('xlabel') or (x_col if x_col else "")
                ylabel = request.form.get('ylabel') or (y_col if y_col else "")
                title = request.form.get('title')

                x_rot = int(request.form.get('x_rot') or 0)
                y_rot = int(request.form.get('y_rot') or 0)

                color = (request.form.get('color') or "").strip() or None
                grid_opt = request.form.get('grid') == 'on'

                fig, ax = plt.subplots(figsize=(10, 6))

                # Build Seaborn arguments dynamically
                box_kwargs = {"data": df, "ax": ax}
                if x_col:
                    box_kwargs["x"] = x_col
                if y_col:
                    box_kwargs["y"] = y_col
                if color:
                    box_kwargs["color"] = color

                sns.boxplot(**box_kwargs)

                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                if title:
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

                ax.tick_params(axis='x', rotation=x_rot)
                ax.tick_params(axis='y', rotation=y_rot)

                if grid_opt:
                    ax.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()

                filename = f"box_plot_{uuid.uuid4().hex[:8]}.jpeg"
                save_path = os.path.join("static", "charts", filename)
                os.makedirs(os.path.join("static", "charts"), exist_ok=True)

                plt.savefig(save_path, format="jpeg", dpi=300, bbox_inches='tight')
                plt.close(fig)

                chart_image = filename
                flash("Chart created successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "visualization/box_plot.html",
        columns=columns,
        chart_image=chart_image,
        available_dfs=available_dfs,
        current_selection=current_selection,
    )

# 3.3 Descriptive Statistics menu
@app.route("/descriptive")
def descriptive_menu():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))
    return render_template("descriptive/descriptive_index.html")

# 3.3.1 Central Tendency (Merged with CDF_DL)
@app.route("/descriptive/central_tendency", methods=["GET", "POST"])
def central_tendency():
    if 'filename' not in session:
        flash('No file uploaded.')
        return redirect(url_for('upload_file'))
    
    # 1. Determine Selection (Default to Main)
    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    # 2. Load the SELECTED Dataframe (Main or Custom)
    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.")
        return redirect(url_for('descriptive_stats_menu'))

    # 3. Get Numeric Columns for the dropdown
    # (Don't rely on session['numeric_columns'] because the custom DF might have different cols)
    numeric_df = df.select_dtypes(include=['number'])
    numeric_columns = numeric_df.columns.tolist()
    
    result = None
    available_dfs = get_df_dict().keys()

    if request.method == "POST":
        action = request.form.get('action') # 'calculate' or 'change_df' (from dropdown)

        if action == 'calculate':
            column = request.form.get('column')
            
            if column in numeric_columns:
                # Calculate ALL 11 stats
                col_data = df[column] # Use the loaded DF
                
                stats = {
                    'Count': col_data.count(),
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Mode': col_data.mode().iloc[0] if not col_data.mode().empty else "No mode",
                    'Min': col_data.min(),
                    '25th Percentile': col_data.quantile(0.25),
                    '50th Percentile': col_data.quantile(0.50),
                    '75th Percentile': col_data.quantile(0.75),
                    'Max': col_data.max(),
                    'Variance': col_data.var(),
                    'Skewness': col_data.skew(),
                    'Std Dev': col_data.std()
                }
                
                # Generate HTML Table
                table_rows = ""
                for measure, value in stats.items():
                    # Format value if it's a number, otherwise just string
                    fmt_val = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    table_rows += f"<tr><td style='padding:8px; border-bottom:1px solid #eee;'><b>{measure}</b></td><td style='padding:8px; border-bottom:1px solid #eee; text-align:right;'>{fmt_val}</td></tr>"
                
                result = f"""
                <table style='width:100%; border-collapse: collapse; margin-top: 1rem;'>
                    <thead>
                        <tr style='background: linear-gradient(135deg, #8b5cf6, #a78bfa); color: white;'>
                            <th style='padding: 0.8rem; text-align: left;'>Measure</th>
                            <th style='padding: 0.8rem; text-align: right;'>Value</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
                """

    return render_template("descriptive/central_tendency.html", 
                         numeric_columns=numeric_columns, 
                         result=result,
                         available_dfs=available_dfs,
                         current_selection=current_selection)

# 3.3.2 Correlation/Pairplot
@app.route("/descriptive/correlation", methods=["GET", "POST"])
def correlation_pairplot():
    if 'filename' not in session:
        flash('No file uploaded.')
        return redirect(url_for('upload_file'))
    
    result_html = None
    plot_url = None
    
    if request.method == "POST":
        action = request.form.get('action')
        
        # Reload DF (full dataset)
        if session['filename'].endswith('csv'):
            df = pd.read_csv(session['filepath'])
        else:
            df = pd.read_excel(session['filepath'])
            
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            flash("No numeric columns found for correlation.")
            return redirect(request.url)

        if action == "correlation":
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            # Convert to HTML table with styling
            result_html = corr_matrix.to_html(classes='table table-striped', float_format='%.3f')
            
        elif action == "pairplot":
            # Generate Pairplot
            plt.figure(figsize=(10, 8)) # Default size
            # Use Sample if >1000 rows (too slow otherwise)
            plot_df = numeric_df.sample(n=min(1000, len(numeric_df))) if len(numeric_df) > 1000 else numeric_df
            
            sns.pairplot(plot_df, height=2.5) # Adjust height per subplot
            
            # Save to string buffer
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close() # Close plot to free memory

    return render_template("descriptive/correlation_pairplot.html", 
                         result_html=result_html, 
                         plot_url=plot_url)

# 3.5 INFERENTIAL STATISTICS
@app.route("/inferential")
def inferential_statistics_menu():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))
    return render_template("inferential/inferential_index.html")

import scipy.stats as st
import numpy as np

# 3.5.1 Confidence Intervals
@app.route("/inferential/confidence_intervals", methods=["GET", "POST"])
def confidence_interval():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('inferential_statistics_menu'))

    # Only pass numeric columns to the template
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    available_dfs = get_df_dict().keys()
    
    result = None

    if request.method == "POST":
        if request.form.get('action') == 'calculate':
            try:
                col = request.form.get('column')
                if not col:
                    raise ValueError("Please select a numeric column.")
                
                conf_level_input = request.form.get('confidence_level')
                if not conf_level_input:
                    raise ValueError("Please enter a confidence level.")
                
                # Convert confidence level to decimal (e.g., 95 -> 0.95)
                conf_level = float(conf_level_input) / 100.0
                if not (0 < conf_level < 1):
                    raise ValueError("Confidence level must be strictly between 0 and 100.")

                # Extract the data and drop missing values
                data = df[col].dropna()
                n = len(data)
                
                if n < 2:
                    raise ValueError("Not enough valid data points in the selected column to calculate a confidence interval.")

                mean = np.mean(data)
                sem = st.sem(data) # Standard error of the mean
                
                # Calculate the confidence interval using the t-distribution
                ci = st.t.interval(confidence=conf_level, df=n-1, loc=mean, scale=sem)

                result = {
                    'column': col,
                    'confidence_level': f"{float(conf_level_input)}%",
                    'sample_size': n,
                    'mean': round(mean, 4),
                    'lower_bound': round(ci[0], 4),
                    'upper_bound': round(ci[1], 4)
                }

                flash("Confidence Interval calculated successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "inferential/confidence_intervals.html",
        numeric_cols=numeric_cols,
        result=result,
        available_dfs=available_dfs,
        current_selection=current_selection,
    )

# 3.5.2 Hypothesis Test (One Sample)
@app.route("/inferential/hypothesis_one_sample", methods=["GET", "POST"])
def hypothesis_one_sample():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('inferential_statistics_menu'))

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    available_dfs = get_df_dict().keys()
    
    result = None

    if request.method == "POST":
        if request.form.get('action') == 'test':
            try:
                # 1. Get Text Inputs
                h0_text = request.form.get('h0_text')
                h1_text = request.form.get('h1_text')
                
                # 2. Get Significance Level
                alpha_str = request.form.get('alpha')
                if not alpha_str:
                    raise ValueError("Please enter a significance level (\u03b1).")
                alpha = float(alpha_str)
                if not (0 < alpha < 1):
                    raise ValueError("Significance level (\u03b1) must be strictly between 0 and 1.")

                # 3. Get Column
                col = request.form.get('column')
                if not col:
                    raise ValueError("Please select a numeric column.")

                # 4. Get Hypothesized Mean
                hypo_mean_str = request.form.get('hypo_mean')
                if not hypo_mean_str:
                    raise ValueError("Please enter a hypothesized mean.")
                hypo_mean = float(hypo_mean_str)

                # Extract data
                data = df[col].dropna()
                if len(data) < 2:
                    raise ValueError("Not enough valid data points to perform the test.")

                # Perform the one-sample t-test
                from scipy import stats
                test_result = stats.ttest_1samp(data, popmean=hypo_mean)
                
                # Unpack results safely handling older and newer versions of scipy
                t_stat = getattr(test_result, 'statistic', test_result[0])
                p_val = getattr(test_result, 'pvalue', test_result[1])
                
                # Handle scalar conversions if arrays are returned
                t_stat = float(np.ravel(t_stat)[0])
                p_val = float(np.ravel(p_val)[0])

                # Determine Conclusion
                if p_val < alpha:
                    conclusion = "Hence, there is enough evidence to reject H<sub>0</sub>."
                    concl_color = "#d32f2f" # Red/Purple for rejection
                else:
                    conclusion = "Hence, there is not enough evidence to reject H<sub>0</sub>. You can try with lesser confidence."
                    concl_color = "#388e3c" # Green for failing to reject

                result = {
                    'h0': h0_text,
                    'h1': h1_text,
                    'alpha': alpha,
                    'column': col,
                    'hypo_mean': hypo_mean,
                    'sample_mean': round(np.mean(data), 4),
                    't_stat': round(t_stat, 4),
                    'p_value': round(p_val, 4),
                    'conclusion': conclusion,
                    'concl_color': concl_color
                }

                flash("Hypothesis Test completed successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "inferential/hypothesis_one_sample.html",
        numeric_cols=numeric_cols,
        result=result,
        available_dfs=available_dfs,
        current_selection=current_selection,
    )

# 3.5.3 Hypothesis Test (Two Sample)
@app.route("/inferential/hypothesis_two_sample", methods=["GET", "POST"])
def hypothesis_two_sample():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))

    # Load custom dataframe lists
    available_dfs = list(get_df_dict().keys())
    
    # Selection states
    selection1 = request.args.get('selected_df1', 'Main Uploaded File')
    selection2 = request.args.get('selected_df2', 'Main Uploaded File')
    
    if request.method == "POST":
        selection1 = request.form.get('selected_df1', 'Main Uploaded File')
        selection2 = request.form.get('selected_df2', 'Main Uploaded File')

    # Load both dataframes
    df1, _ = load_selected_df(selection1)
    df2, _ = load_selected_df(selection2)
    
    if df1 is None or df2 is None:
        flash("Error loading dataframes.", "error")
        return redirect(url_for('inferential_statistics_menu'))

    # Get numeric columns for both
    numeric_cols1 = df1.select_dtypes(include='number').columns.tolist()
    numeric_cols2 = df2.select_dtypes(include='number').columns.tolist()
    
    result = None

    if request.method == "POST":
        if request.form.get('action') == 'test':
            try:
                # 1. Hypothesis inputs
                h0_text = request.form.get('h0_text')
                h1_text = request.form.get('h1_text')
                
                # 2. Significance Level
                alpha_str = request.form.get('alpha')
                if not alpha_str:
                    raise ValueError("Please enter a significance level (\u03b1).")
                alpha = float(alpha_str)
                if not (0 < alpha < 1):
                    raise ValueError("Significance level (\u03b1) must be strictly between 0 and 1.")

                # 3. Columns
                col1 = request.form.get('column1')
                col2 = request.form.get('column2')
                if not col1 or not col2:
                    raise ValueError("Please select columns from both data frames.")

                # 4. Variance & Alternative
                equal_var_opt = request.form.get('equal_var') == 'True'
                alternative = request.form.get('alternative', 'two-sided')

                # Extract and clean data
                data1 = df1[col1].dropna()
                data2 = df2[col2].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    raise ValueError("Not enough valid data points in one or both columns.")

                test_result = st.ttest_ind(
                    data1, 
                    data2, 
                    equal_var=equal_var_opt, 
                    alternative=alternative
                )

                t_stat = getattr(test_result, 'statistic', test_result[0])
                p_val = getattr(test_result, 'pvalue', test_result[1])
                
                t_stat = float(np.ravel(t_stat)[0])
                p_val = float(np.ravel(p_val)[0])

                if p_val < alpha:
                    conclusion = "Hence, there is enough evidence to reject H<sub>0</sub>."
                    concl_color = "#388e3c"
                else:
                    conclusion = "Hence, there is not enough evidence to reject H<sub>0</sub>. You can try with lesser confidence."
                    concl_color = "#d32f2f"

                result = {
                    'h0': h0_text,
                    'h1': h1_text,
                    'alpha': alpha,
                    'col1': f"{selection1} -> {col1}",
                    'col2': f"{selection2} -> {col2}",
                    'mean1': round(np.mean(data1), 4),
                    'mean2': round(np.mean(data2), 4),
                    'equal_var': equal_var_opt,
                    'alternative': alternative,
                    't_stat': round(t_stat, 4),
                    'p_value': round(p_val, 4),
                    'conclusion': conclusion,
                    'concl_color': concl_color
                }

                flash("Hypothesis Test completed successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "inferential/hypothesis_two_sample.html",
        numeric_cols1=numeric_cols1,
        numeric_cols2=numeric_cols2,
        result=result,
        available_dfs=available_dfs,
        selection1=selection1,
        selection2=selection2
    )

# 3.6 LINEAR REGRESSION MODEL UNIT
@app.route("/regression")
def linear_regression_menu():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))
    
    return render_template("regression/regression_index.html")

# Dictionary to store trained models in memory
trained_models = {}

# 3.6.1.1 Train Prediction Model
@app.route("/regression/train_prediction_model", methods=["GET", "POST"])
def train_prediction_model():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))

    current_selection = request.args.get('selected_df', 'Main Uploaded File')
    if request.method == "POST":
        current_selection = request.form.get('selected_df', 'Main Uploaded File')

    df, path = load_selected_df(current_selection)
    if df is None:
        flash("Error loading dataframe.", "error")
        return redirect(url_for('linear_regression_menu'))

    # Dropdowns: Y must be numeric, X can be any (except Y)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()
    available_dfs = get_df_dict().keys()
    
    result = None

    if request.method == "POST":
        if request.form.get('action') == 'train':
            try:
                # 1. Model Name
                model_name = request.form.get('model_name')
                if not model_name:
                    raise ValueError("Please provide a name for the model.")
                if model_name in trained_models:
                    raise ValueError("A model with this name already exists. Please choose a unique name.")

                # 2. Dependent Variable (Y)
                y_col = request.form.get('y_col')
                if not y_col:
                    raise ValueError("Please select a Dependent Variable (Y).")

                # 3. Independent Variables (X)
                x_cols = request.form.getlist('x_cols')
                if not x_cols:
                    raise ValueError("Please select at least one Independent Variable (X).")
                if y_col in x_cols:
                    raise ValueError("The Dependent Variable (Y) cannot be in the Independent Variables (X).")

                # 4. Train-Test Split
                test_size_str = request.form.get('test_size')
                if not test_size_str:
                    raise ValueError("Please select a train-test split.")
                test_size = float(test_size_str)

                # Prepare Data
                model_data = df[[y_col] + x_cols].dropna()
                if len(model_data) < 10:
                    raise ValueError("Not enough valid rows to train a model after dropping missing values.")

                Y = model_data[y_col]
                X_raw = model_data[x_cols]

                # Check if X contains non-numeric (categorical)
                is_all_numeric = all(pd.api.types.is_numeric_dtype(X_raw[c]) for c in x_cols)
                
                if is_all_numeric:
                    X = sm.add_constant(X_raw)
                else:
                    # Convert categorical to dummies
                    X_dummies = pd.get_dummies(X_raw, drop_first=True, dtype=int)
                    X = sm.add_constant(X_dummies)

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

                # Train Model
                model = sm.OLS(y_train, X_train).fit()

                # Model Evaluation
                predictions = model.predict(X_test)
                residuals = y_test - predictions
                mae = residuals.abs().mean()

                # Save Model for later prediction
                trained_models[model_name] = {
                    'model': model,
                    'features': X.columns.tolist() # Store exact dummy columns needed later
                }

                # Extract Summary Table HTML [web:159]
                summary_html = model.summary().tables[1].as_html()

                result = {
                    'model_name': model_name,
                    'y_col': y_col,
                    'r_squared': round(model.rsquared, 4),
                    'adj_r_squared': round(model.rsquared_adj, 4),
                    'train_size': len(y_train),
                    'test_size': len(y_test),
                    'mae': round(mae, 4),
                    'summary_html': summary_html
                }

                flash(f"Model '{model_name}' trained successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "regression/cross_sectional/train_cs.html",
        numeric_cols=numeric_cols,
        all_cols=all_cols,
        result=result,
        available_dfs=available_dfs,
        current_selection=current_selection
    )

# 3.6.1.2 Make Predictions
@app.route("/regression/make_predictions", methods=["GET", "POST"])
def make_predictions():
    if 'filename' not in session:
        flash('No file uploaded. Please upload first.')
        return redirect(url_for('upload_file'))

    # Access the global dictionary where we stored the trained models
    available_models = list(trained_models.keys())
    
    result = None

    if request.method == "POST":
        if request.form.get('action') == 'predict':
            try:
                # 1. Selected Model
                model_name = request.form.get('model_name')
                if not model_name or model_name not in trained_models:
                    raise ValueError("Please select a valid trained model.")

                # 2. Extract predictors input
                predictors_str = request.form.get('predictors')
                if not predictors_str:
                    raise ValueError("Please enter the predictor values.")

                # Retrieve the saved model and features
                saved_data = trained_models[model_name]
                model = saved_data['model']
                expected_features = saved_data['features'] 
                
                has_const = 'const' in expected_features
                num_expected_vars = len(expected_features) - (1 if has_const else 0)

                # Parse input: split by semicolon for multiple rows, then by comma for values
                try:
                    # E.g., "1, 2; 3, 4" -> [[1.0, 2.0], [3.0, 4.0]]
                    raw_rows = predictors_str.split(';')
                    parsed_rows = []
                    
                    for row in raw_rows:
                        if not row.strip():
                            continue # Skip empty segments
                        row_vals = [float(x.strip()) for x in row.split(',')]
                        
                        if len(row_vals) != num_expected_vars:
                            feature_names = [f for f in expected_features if f != 'const']
                            raise ValueError(f"Each row must have exactly {num_expected_vars} predictors ({', '.join(feature_names)}). Found {len(row_vals)} in row: '{row}'.")
                        
                        if has_const:
                            row_vals.insert(0, 1.0)
                            
                        parsed_rows.append(row_vals)
                        
                    if not parsed_rows:
                        raise ValueError("No valid prediction rows found.")
                        
                except ValueError as ve:
                    # If it's our custom ValueError from above, raise it directly
                    if "must have exactly" in str(ve) or "No valid" in str(ve):
                        raise ve
                    else:
                        raise ValueError("Predictors must be numbers. Use commas to separate values, and semicolons to separate multiple rows.")

                # Convert to a DataFrame
                predict_df = pd.DataFrame(parsed_rows, columns=expected_features)

                # Make the predictions
                predictions = model.predict(predict_df).round(4).tolist()

                # Format the output for display
                # We will zip the original input rows (without const) with their predictions
                display_results = []
                for i, row in enumerate(parsed_rows):
                    # Remove the constant for display if it was added
                    display_vals = row[1:] if has_const else row
                    display_results.append({
                        'inputs': display_vals,
                        'prediction': predictions[i]
                    })

                result = {
                    'model_name': model_name,
                    'predictions_list': display_results
                }

                flash("Predictions generated successfully!", "success")

            except Exception as e:
                flash(f"Error: {str(e)}", "error")

    return render_template(
        "regression/cross_sectional/predict_cs.html",
        available_models=available_models,
        result=result
    )

#end...
if __name__ == "__main__":
    app.run(debug=True)

