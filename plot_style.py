#import all the libraries needed
from import_dep import *


def set_plot_style(export_data = False, powerpoint_data = False, use_tex=True):
    """
    Set publication-quality plot styles.
    
    Parameters:
    -----------
    use_tex : bool
        Whether to use LaTeX for rendering text (default: True)
    """
    
    # Set the figure size based on whether we are visualising or exporting the data
    if export_data == True:
        fig_size = [3.5, 2.625] # Publication ready sizes
    else:
        fig_size = [9, 6] # Better for visualisation
    
    # Use a colorblind-friendly colormap with at least 10 distinct colors
    cmap_colors =   sns.color_palette("colorblind", 12) #sns.color_palette("bright", 10)
    color_cycler = cycler('color', cmap_colors)
    color_cycler_2 = cycler('color', ['#0C5DA5', '#00B945', '#FF9500', 
                                           '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    
    # Science style settings
    plt.rcParams.update({
        # Figure settings
        'figure.figsize':fig_size,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.constrained_layout.use': False, # Enable constrained layout by default
        
        # Font and text settings
        'font.family': ['serif'],
        'font.size': 9,  # Base font size
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'mathtext.fontset': 'dejavuserif',
        'text.usetex': use_tex,
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        
        # Axes settings
        'axes.linewidth': 0.5,
        'axes.prop_cycle': color_cycler ,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.4,
        
        # Line settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4.0,
        # Errorbar settings
        'errorbar.capsize': 0,
        
        # Tick settings
        'xtick.direction': 'in',
        'xtick.major.size': 3.0,
        'xtick.major.width': 0.5,
        'xtick.minor.size': 1.5,
        'xtick.minor.visible': True,
        'xtick.minor.width': 0.5,
        'xtick.top': True,
        
        'ytick.direction': 'in',
        'ytick.major.size': 3.0,
        'ytick.major.width': 0.5,
        'ytick.minor.size': 1.5,
        'ytick.minor.visible': True,
        'ytick.minor.width': 0.5,
        'ytick.right': True,
        
        # Prevent autolayout to ensure that the figure size is obeyed
        #'figure.autolayout': False,
    })
    
    return fig_size


# Define a function for the PPTX module to add a slide with a title and image
# --- Define the standard font for the slide (consistent usage) ---
SLIDE_FONT = "Avenir" # Note: Ensure this font is available on the system running the code AND the system viewing the PPTX

def add_slide(fig,
              title,
              notes,
              prs,
              path_out,
              layout: str = 'Vertical' # 'Horizontal' for text to right of figure or 'Vertical' for text below figure
              ):
    """
    Adds a slide to the presentation with a custom title bar (gradient),
    figure, and notes, using the defined SLIDE_FONT. Layout can be 'Horizontal'
    (figure left, text right) or 'Vertical' (figure top, text bottom).

    Args:
        fig: The matplotlib figure object to add.
        title (str): The title for the slide.
        notes (list): A list of strings for the notes section.
        prs: The Presentation object.
        path_out (str or Path): The directory to save the temporary image file.
        layout (str): 'Horizontal' or 'Vertical'. Defaults to 'Horizontal'.
    """
    # Check if figure exists and is valid
    if fig is None or not hasattr(fig, 'axes') or len(fig.axes) == 0:
        print(f"Skipping '{title}': Figure does not exist or is empty.")
        if fig:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError: pass
            except Exception as e_close: print(f"Error closing invalid figure '{title}': {e_close}")
        return

    # --- Use BLANK layout ---
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # --- Set background color ---
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(240, 240, 240)

    # --- Create Title Rectangle ---
    slide_width = prs.slide_width
    slide_height = prs.slide_height
    title_box_height = Inches(0.819)
    title_box = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        left=0, top=0, width=slide_width, height=title_box_height
    )
    # Remove border
    title_box.line.fill.background()
    title_box.line.width = Pt(0)

    # --- Apply Linear Gradient Fill to Title Rectangle ---
    fill = title_box.fill
    fill.gradient()
    fill.gradient_angle = 0.0 # Linear Left to Right

    # --- Define gradient stops ---
    if len(fill.gradient_stops) >= 2: # Check if stops exist
        stop1 = fill.gradient_stops[0]
        stop1.color.rgb = RGBColor(40, 80, 160) # Left color
        stop1.position = 0.1

        stop2 = fill.gradient_stops[1]
        stop2.color.rgb = RGBColor(100, 160, 220) # Right color
        stop2.position = 1.0
    else:
        print(f"Warning: Could not set gradient stops for title '{title}'. Default gradient may apply.")
        # Fallback: fill.solid(); fill.fore_color.rgb = RGBColor(68, 114, 196)

    # --- Add Text Box FOR the title ON TOP ---
    title_left = Inches(0.2)
    title_top = Inches(0.1)
    title_width = slide_width - Inches(0.4)
    title_textbox_height = title_box_height - Inches(0.1)
    title_textbox = slide.shapes.add_textbox(
        title_left, title_top, title_width, title_textbox_height
    )
    tf = title_textbox.text_frame
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = tf.paragraphs[0]
    p.text = title
    p.font.name = SLIDE_FONT
    p.font.size = Pt(20)
    p.font.bold = False
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.LEFT

    # --- Save the figure as an image ---
    output_dir = Path(path_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    img_path = output_dir / f'{safe_title}.png'

    try:
        fig.savefig(img_path, dpi=300, bbox_inches='tight', transparent=True)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError: print("Warning: Matplotlib not imported, cannot close figure.")
        except Exception as e_close: print(f"Error closing figure '{title}': {e_close}")
    except Exception as e_save:
        print(f"Error saving figure '{title}': {e_save}")
        if fig:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception: pass
        return

    # --- Define Layout Areas ---
    margin = Inches(0.2)
    content_top = title_box_height + margin
    content_height = slide_height - content_top - margin
    total_content_width = slide_width - (2 * margin) # Total width available for content

    if layout.lower() == 'vertical':
        # Vertical Layout: Figure top (60%), Text bottom (40%)
        top_section_height = 0.6 * content_height
        bottom_section_height = 0.4 * content_height
        bottom_section_top = content_top + top_section_height + margin

        # Figure area
        img_left = margin
        img_top = content_top
        max_img_width = total_content_width
        max_img_height = top_section_height
        img_center_ref_width = total_content_width # Width to center the image within

        # Notes area
        notes_left = margin
        notes_top = bottom_section_top
        notes_width = total_content_width
        notes_height = bottom_section_height

    else: # Default to Horizontal Layout
        
        if layout.lower() != 'horizontal':
            print(f"Warning: Invalid layout '{layout}' specified. Defaulting to 'Horizontal'.")
            
        # Horizontal Layout: Figure left (60%), Text right (40%)
        # Calculate total width available for columns after accounting for 3 margins (left, center, right)
        available_columns_width = slide_width - (3 * margin)
        # Left column takes 60% of the available width
        left_column_width = 0.6 * available_columns_width
        # Right column takes 40% of the available width
        right_column_width = 0.4 * available_columns_width
        # Right column starts after the left margin, the left column width, and the center margin
        right_column_left = margin + left_column_width + margin

        # Figure area
        img_left = margin
        img_top = content_top
        max_img_width = left_column_width
        max_img_height = content_height
        img_center_ref_width = left_column_width # Width to center the image within

        # Notes area
        notes_left = right_column_left
        notes_top = content_top
        notes_width = right_column_width
        notes_height = content_height
        
        

    try:
        # Add picture, initially using max width for its area
        pic = slide.shapes.add_picture(str(img_path), int(img_left), int(img_top), width=int(max_img_width))

        # --- Scaling and Centering Logic ---
        scaled_height = False
        scaled_width = False

        # 1. Scale height if it exceeds max height, adjusting width proportionally
        if pic.height > max_img_height:
            scale_factor = max_img_height / pic.height
            pic.height = int(max_img_height)
            pic.width = int(pic.width * scale_factor)
            scaled_height = True

        # 2. Scale width if it exceeds max width (could happen after height scaling)
        if pic.width > max_img_width:
            scale_factor = max_img_width / pic.width
            pic.width = int(max_img_width)
            # Only adjust height if width scaling happened *after* height scaling didn't max it out
            if not scaled_height:
                 pic.height = int(pic.height * scale_factor)
            scaled_width = True

        # 3. Center the potentially scaled image within its designated area
        # Center vertically
        if pic.height < max_img_height:
            pic.top = int(img_top + (max_img_height - pic.height) / 2)
        else: # If scaling maxed out height, ensure it's at the top
            pic.top = int(img_top)

        # Center horizontally within the reference width for the layout
        if pic.width < img_center_ref_width:
            pic.left = int(img_left + (img_center_ref_width - pic.width) / 2)
        else: # If scaling maxed out width, ensure it's at the left edge
             pic.left = int(img_left)


        print(f"Added slide: {title} (Layout: {layout})")

        # --- Add Notes Box if notes are provided ---
        if notes and isinstance(notes, list) and any(n and n.strip() for n in notes):
            # Ensure notes box dimensions are integers
            notes_box = slide.shapes.add_textbox(int(notes_left), int(notes_top), int(notes_width), int(notes_height))
            notes_tf = notes_box.text_frame
            notes_tf.word_wrap = True
            notes_tf.vertical_anchor = MSO_ANCHOR.TOP # Anchor text to top
            notes_tf.clear()

            for line in notes:
                line = line.strip()
                if line:
                    p_notes = notes_tf.add_paragraph()
                    # Add bullet point using unicode character
                    p_notes.text = f"• {line}"
                    p_notes.font.name = SLIDE_FONT
                    p_notes.font.size = Pt(14)
                    p_notes.font.color.rgb = RGBColor(0, 0, 0)
                    p_notes.line_spacing = 1.1
                    p_notes.space_before = Pt(3) # Add space before bullet point
                    p_notes.space_after = Pt(3) # Add space after bullet point

            # Remove extra space after the last paragraph
            if len(notes_tf.paragraphs) > 0:
                notes_tf.paragraphs[-1].space_after = Pt(0)

    except FileNotFoundError:
        print(f"Error adding picture: Image file not found at '{img_path}'.")
    except Exception as e_add:
        print(f"Error adding picture or notes for slide '{title}': {e_add}")

    # Clean up the saved image file
    try:
        os.remove(img_path)
    except OSError as e_remove:
        print(f"Warning: Could not remove temporary image file '{img_path}': {e_remove}")
